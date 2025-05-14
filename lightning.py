import torch
import torchaudio
from cosine import WarmupCosineScheduler
from datamodule.transforms import TextTransform

from espnet.nets.batch_beam_search import BatchBeamSearch
from espnet.nets.pytorch_backend.e2e_asr_conformer import E2E
from espnet.nets.scorers.length_bonus import LengthBonus
from espnet.nets.scorers.bridged_lm_scorer import BridgedLMScorer
from pytorch_lightning import LightningModule


def compute_word_level_distance(seq1, seq2):
    seq1, seq2 = seq1.lower().split(), seq2.lower().split()
    return torchaudio.functional.edit_distance(seq1, seq2)


class ModelModule(LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.save_hyperparameters(args)

        self.modality = args.modality
        self.text_transform = TextTransform()
        self.token_list = self.text_transform.token_list

        self.model = E2E(len(self.token_list), self.modality, ctc_weight=getattr(args, "ctc_weight", 0.1)).to('cuda')
        if getattr(args, "freeze_layers", None):
            for module_to_freeze in [
                self.model.frontend,
                self.model.proj_encoder,
                # self.model.encoder,  # don't freeze whole layer. Just the MLP at the end of it
            ]:
                for param in module_to_freeze.parameters():
                    param.requires_grad = False
            print(f"Layers frozen")

        # -- initialise
        if getattr(args, "checkpoint_path", None):
            ckpt = torch.load(args.checkpoint_path, map_location=lambda storage, loc: storage)
            
            state_dict = (
                ckpt.get("state_dict")
                or ckpt.get("model_state_dict")
                or ckpt 
            )
            state_dict = {
                k.replace("model.", "", 1): v
                for k, v in state_dict.items()
            }

            if getattr(args, "transfer_frontend", False):
                tmp = {k: v for k, v in state_dict.items()
                    if k.startswith(("trunk.", "frontend3D."))}
                self.model.frontend.load_state_dict(tmp)

            elif getattr(args, "transfer_encoder", False):
                self.model.frontend.load_state_dict(
                    {k.replace("frontend.", ""): v for k, v in state_dict.items()
                    if k.startswith("frontend.")})
                self.model.proj_encoder.load_state_dict(
                    {k.replace("proj_encoder.", ""): v for k, v in state_dict.items()
                    if k.startswith("proj_encoder.")})
                self.model.encoder.load_state_dict(
                    {k.replace("encoder.", ""): v for k, v in state_dict.items()
                    if k.startswith("encoder.")})

            else:
                self.model.load_state_dict(state_dict, strict=True)


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay, betas=(0.9, 0.98))
        scheduler = WarmupCosineScheduler(optimizer, self.args.warmup_epochs, self.args.epochs, len(self.trainer.datamodule.train_dataloader()) / self.trainer.num_devices / self.trainer.num_nodes)
        scheduler = {"scheduler": scheduler, "interval": "step"}
        return [optimizer], [scheduler]

    def forward(self, sample):
        self.beam_search = get_beam_search_decoder(
            self.model,
            self.token_list,
            beam_size=self.args.beam_size,
            ctc_weight=self.args.ctc_weight,
            lm_weight=self.args.lm_weight,
            lm_name=self.args.lm_name,
        )
        x = self.model.frontend(sample.to('cuda').unsqueeze(0))  # batchify before passing
        x = self.model.proj_encoder(x)
        enc_feat, _ = self.model.encoder(x, None)
        enc_feat = enc_feat.squeeze(0)
        nbest_hyps = self.beam_search(enc_feat)
        nbest_hyps = [h.asdict() for h in nbest_hyps[: min(len(nbest_hyps), 1)]]
        predicted_token_id = torch.tensor(list(map(int, nbest_hyps[0]["yseq"][1:])))
        predicted = self.text_transform.post_process(predicted_token_id).replace("<eos>", "")
        return predicted

    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, step_type="val")

    def test_step(self, sample, sample_idx):
        # pull out the raw input tensor
        inp = sample["input"]  # shape: (C, T, H, W) for video, or (T, C) for audio

        # forward through frontend+proj_encoder+encoder
        x = self.model.frontend(inp.unsqueeze(0))      # (1, T, D)
        x = self.model.proj_encoder(x)                 # (1, T, D')
        enc_feat, _ = self.model.encoder(x, None)       # (1, T, D'')
        enc_feat = enc_feat.squeeze(0)                  # (T, D'')

        # 2) pull CTC logits & look at the very first frame
        with torch.no_grad():
            ctc_logits = self.model.ctc.log_softmax(enc_feat.unsqueeze(0))  # (1, T, V)
        c0 = ctc_logits[0, 0]  # logits at time 0, shape (V,)
        top5 = torch.topk(c0, 5).indices.cpu().tolist()

        nbest_hyps = self.beam_search(enc_feat)  # NOTE: not using batch beam search
        nbest_hyps = [h.asdict() for h in nbest_hyps[: min(len(nbest_hyps), 1)]]  # single best hyp
        predicted_token_id = torch.tensor(list(map(int, nbest_hyps[0]["yseq"][1:])))
        predicted = self.text_transform.post_process(predicted_token_id).replace("<eos>", "")

        actual_token_id = sample["target"]
        actual = self.text_transform.post_process(actual_token_id)

        # inside test_step, after you compute `predicted` and `actual`:
        print(f"[ACTUAL {sample_idx}]: '{actual}'", flush=True)
        print(f"[PRED   {sample_idx}]: '{predicted}'", flush=True)

        self.total_edit_distance += compute_word_level_distance(actual, predicted)
        self.total_length += len(actual.split())
        # print("Transcript was compared against ground truth label")

        return

    def training_step(self, batch, batch_idx):
        loss = self._step(batch, batch_idx, "train")
        # batch_size = batch["inputs"].size(0)
        # batch_sizes = self.all_gather(batch_size)
        # loss *= batch_sizes.size(0) / batch_sizes.sum()  # world size / batch size

        self.log("monitoring_step", torch.tensor(self.global_step, dtype=torch.float32))

        return loss

    def _step(self, batch, batch_idx, step_type):
        loss, loss_ctc, loss_att, acc = self.model(batch["inputs"], batch["input_lengths"], batch["targets"])
        batch_size = len(batch["inputs"])

        if step_type == "train":
            self.log("loss", loss, on_step=True, on_epoch=True, batch_size=batch_size)
            self.log("loss_ctc", loss_ctc, on_step=False, on_epoch=True, batch_size=batch_size, sync_dist=True)
            self.log("loss_att", loss_att, on_step=False, on_epoch=True, batch_size=batch_size, sync_dist=True)
            self.log("decoder_acc", acc, on_step=True, on_epoch=True, batch_size=batch_size, sync_dist=True)
        else:
            self.log("loss_val", loss, batch_size=batch_size, sync_dist=True)
            self.log("loss_ctc_val", loss_ctc, batch_size=batch_size, sync_dist=True)
            self.log("loss_att_val", loss_att, batch_size=batch_size, sync_dist=True)
            self.log("decoder_acc_val", acc, batch_size=batch_size, sync_dist=True)

        if step_type == "train":
            self.log("monitoring_step", torch.tensor(self.global_step, dtype=torch.float32))

        return loss

    def on_test_epoch_start(self):
        self.total_length = 0
        self.total_edit_distance = 0
        self.text_transform = TextTransform()
        self.beam_search = get_beam_search_decoder(
            self.model,
            self.token_list,
            beam_size=self.args.beam_size,
            ctc_weight=self.args.ctc_weight,
            lm_weight=self.args.lm_weight,
            lm_name=self.args.lm_name,
        )

    def on_test_epoch_end(self):
        self.log("wer", self.total_edit_distance / self.total_length)


def get_beam_search_decoder(
    model,
    token_list,
    penalty=0,
    ctc_weight=0.1,
    lm_weight=0.1,
    beam_size=40,
    lm_name="meta-llama/Llama-3.2-11B-Vision"
):
    sos = model.odim - 1
    eos = model.odim - 1
    scorers = model.scorers()

    print(f"ctc_weight: {ctc_weight}")
    if lm_weight > 0:
        print(f"lm_weight: {lm_weight}")
        scorers["lm"] = BridgedLMScorer(token_list, model_name=lm_name)
    scorers["length_bonus"] = LengthBonus(len(token_list))
    beta = (1 - lm_weight) / 10
    weights = {
        "decoder": 9 * beta,
        "ctc": beta,
        "lm": lm_weight,
        "length_bonus": penalty,
    }

    return BatchBeamSearch(
        beam_size=beam_size,
        vocab_size=len(token_list),
        weights=weights,
        scorers=scorers,
        sos=sos,
        eos=eos,
        token_list=token_list,
        pre_beam_score_key=None if ctc_weight == 1.0 else "decoder",
    )
