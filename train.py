import os
import logging
from argparse import ArgumentParser

from datamodule.data_module import DataModule
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint


# Set environment variables and logger level
# logging.basicConfig(level=logging.WARNING)


def get_trainer(args):
    seed_everything(42, workers=True)
    checkpoint = ModelCheckpoint(
        dirpath=os.path.join(args.exp_dir, args.exp_name) if args.exp_dir else None,
        monitor="monitoring_step",
        mode="max",
        save_last=True,
        filename="{epoch}",
        save_top_k=10,
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks = [checkpoint, lr_monitor]

    return Trainer(
        sync_batchnorm=True,
        default_root_dir=args.exp_dir,
        max_epochs=args.epochs,
        num_nodes=args.num_nodes,
        # devices=args.gpus,
        accelerator="gpu",
        # strategy=DDPStrategy(find_unused_parameters=False,),
        callbacks=[c for c in callbacks if not isinstance(c, LearningRateMonitor)],
        reload_dataloaders_every_n_epochs=1,
        logger=False,
        gradient_clip_val=10.0,
    )


def get_lightning_module(args):
    # Set modules and trainer
    from lightning import ModelModule
    modelmodule = ModelModule(args)
    return modelmodule


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--exp-dir",
        type=str,
        default="exp_dir",
        help="Directory to save checkpoints and logs to.",
    )
    parser.add_argument(
        "--exp-name",
        type=str,
        help="Experiment name",
        required=True,
    )
    parser.add_argument(
        "--modality",
        type=str,
        choices=["audio", "video"],
        default="video",
        help="Type of input modality",
    )
    parser.add_argument(
        "--train-root-dir",
        type=str,
        default="trainset",
        help="Root directory of training dataset",
    )
    parser.add_argument(
        "--val-root-dir",
        type=str,
        default="testset",
        help="Root directory of validation dataset",
    )
    parser.add_argument(
        "--train-file",
        type=str,
        default="labels.txt",
        help="Filename of training label list",
    )
    parser.add_argument(
        "--val-file",
        type=str,
        default="labels.txt",
        help="Filename of validation label list",
    )
    parser.add_argument(
        "--num-nodes",
        default=1,
        type=int,
        help="Number of machines used",
    )
    parser.add_argument(
        "--freeze-layers",
        action="store_true",
        help="Boolean. Wheter to freeze parts of the model or not.",
    )
    parser.add_argument(
        "--batch-size",
        default=8,
        type=int,
        help="Batch size",
    )
    parser.add_argument(
        "--gpus",
        default=1,
        type=int,
        help="Number of GPUs in the machine",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        help="Path to the model to be trained",
        required=True,
    )
    parser.add_argument(
        "--warmup-epochs",
        type=int,
        default=2,
        help="Number of epochs for warmup",
    )
    parser.add_argument(
        "--epochs",
        default=5,
        type=int,
        help="Number of epochs",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=1600,
        help="Maximal number of frames in a batch",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-5,
        help="Learning rate",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.03,
        help="Weight decay",
    )
    parser.add_argument(
        "--ctc-weight",
        type=float,
        default=0.1,
        help="CTC weight",
    )
    parser.add_argument(
        "--train-num-buckets",
        type=int,
        default=400,
        help="Bucket size for the training set",
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
    modelmodule = get_lightning_module(args)
    datamodule = DataModule(args, train_num_buckets=args.train_num_buckets, batch_size=args.batch_size)
    trainer = get_trainer(args)
    trainer.fit(model=modelmodule, datamodule=datamodule)

if __name__ == "__main__":
    import torch
    torch.set_float32_matmul_precision("medium")
    torch.cuda.empty_cache()
    cli_main()

# To freeze some layers, pass --freeze-layers
#   To change which layers to freeze, go to lightning __init__

# For multi-GPU training, uncomment lines 107-109 in lightning.py
