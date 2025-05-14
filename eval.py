import logging
from argparse import ArgumentParser

import torch
from datamodule.data_module import DataModule
from pytorch_lightning import Trainer


logging.basicConfig(level=logging.WARNING)


def get_trainer(args):
    return Trainer(num_nodes=1, devices=1, accelerator="gpu")


def get_lightning_module(args):
    from lightning import ModelModule
    modelmodule = ModelModule(args)
    return modelmodule


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--modality",
        type=str,
        choices=["audio", "video"],
        default="video",
        help="Type of input modality",
    )
    parser.add_argument(
        "--test-root-dir",
        type=str,
        default="testset",
        help="Root directory of testing dataset",
    )
    parser.add_argument(
        "--test-file",
        type=str,
        default="labels.txt",
        help="Filename of testing label list",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        help="Path to the model to be tested",
        required=True,
    )
    parser.add_argument(
        "--lm-weight",
        default=0.0,
        type=float,
        help="Weight of the lanugage model for beam search",
    )
    parser.add_argument(
        "--ctc-weight",
        default=0.1,
        type=float,
        help="CTC weight for beam search",
    )
    parser.add_argument(
        "--lm-name",
        default="meta-llama/Llama-3.2-11B-Vision",
        type=str,
        help="HuggingFace ID of the LM",
    )
    parser.add_argument(
        "--beam-size",
        type=int,
        default=20,
        help="Beam size. Larger values give better accuracy but run slower.",
    )
    parser.add_argument(
        "--decode-snr-target",
        type=float,
        default=999999,
        help="Level of signal-to-noise ratio (SNR)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Flag to use debug level for logging",
    )
    return parser.parse_args()


def init_logger(debug):
    fmt = "%(asctime)s %(message)s" if debug else "%(message)s"
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(format=fmt, level=level, datefmt="%Y-%m-%d %H:%M:%S")


def cli_main():
    args = parse_args()
    init_logger(args.debug)
    modelmodule = get_lightning_module(args)
    datamodule = DataModule(args)
    trainer = get_trainer(args)
    trainer.test(model=modelmodule, datamodule=datamodule)


if __name__ == "__main__":
    import torch
    torch.set_float32_matmul_precision("medium")
    torch.cuda.empty_cache()
    cli_main()

# NOTE: to change CTC or CE weights, go to lightning get_beam_search_decoder

# to use LM, pass --lm-weight. to change beam size, pass --beam-size

# The labels file has to be at args.root_dir\labels\args.test_file
#   It must be a text file with the format "dataset_name,rel_path,input_length,token_id"
#   where token_id contains the tokens of the transcript

# The input data must be at args.root_dir\dataset_name\rel_path,
#   where the last two arguments correspond to the label in the text file

# The VSR tokens start with a blank space, many gpt2 tokens also do, so this is good.
#   however, gpt2 tokens are generally shorter

# To change the LM model, go to lightning.py line 164
