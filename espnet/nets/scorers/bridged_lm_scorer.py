import os

import torch, torch.nn.functional as F

from transformers import AutoTokenizer, AutoModelForCausalLM
from espnet.nets.scorer_interface import BatchScorerInterface
from datamodule.transforms import TextTransform


class BridgedLMScorer(BatchScorerInterface):
    def __init__(self, token_list, model_name="meta-llama/Llama-3.2-11B"):
        super().__init__()

        self.tok = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=True,
            trust_remote_code=True
        )
        
        self.tok.pad_token = self.tok.eos_token

        self.vsr_tokens = token_list
        self.vsr_str_tokens = []
        self.text_transform = TextTransform()

        hf_token = os.environ["HUGGINGFACE_HUB_TOKEN"]
        self.lm = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            token=hf_token,
            trust_remote_code=True
        ).eval()
        self.vsr_tokens = token_list

        self.piece_seqs = []
        longest = 0
        for t in token_list:
            if t == "<blk>" or t == "<sos/eos>":

                self.piece_seqs.append([])
                continue
            s = " " + t if t not in " \n\t" else t
            pieces = self.tok(s, add_special_tokens=False)["input_ids"]
            self.piece_seqs.append(pieces)
            longest = max(longest, len(pieces))

        self.piece_ids = torch.full((len(token_list), longest),
                                    fill_value=-1, dtype=torch.long)
        for j, seq in enumerate(self.piece_seqs):
            if seq:
                self.piece_ids[j, :len(seq)] = torch.tensor(seq)
        self.piece_ids = self.piece_ids.to("cuda")


    def _merge_to_vsr(self, logp_lm):
        """
        logp_lm : (B, V_lm)  – log-probs over the LM’s whole vocab
        returns : (B, V_vsr) – log-probs over your lip-reading vocab
        """
        B, Vlm = logp_lm.shape
        idx    = self.piece_ids.unsqueeze(0).expand(B, -1, -1)


        safe_idx = idx.clamp(min=0)
        gathered = torch.gather(
            logp_lm.unsqueeze(1).expand(-1, idx.size(1), -1),
            2,
            safe_idx
        )

        gathered[idx == -1] = float("-inf")

        return torch.logsumexp(gathered, dim=-1)


    def batch_score(self, ys, states, xs):
        # # skip <blank> and <eos>
        # text = ["".join(self.vsr_str_tokens[int(i)] for i in y.tolist()
        #                 if i < len(self.vsr_str_tokens) - 1 and i > 0)
        #         for y in ys]  # List[str]
        # text = [sen if sen else " " for sen in text]  # no sentence must be empty
        text = [
            self.text_transform.post_process(y).replace("<eos>", "") for y in ys
        ]
        text = ["." if sen == "" else sen for sen in text]

        enc = self.tok(text, return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            out = self.lm(input_ids=enc.input_ids,
                        attention_mask=enc.attention_mask)
            logp_lm = F.log_softmax(out.logits[:, -1, :], dim=-1)

        vsr_logp = self._merge_to_vsr(logp_lm)
        return vsr_logp, [None] * ys.size(0)
