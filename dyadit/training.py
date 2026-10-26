"""Training entry point for the main DyaDiT model."""

import argparse
import os
from pathlib import Path

import pytorch_lightning as pl
import yaml
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from .checkpoint import load_system_weights
from .config import load_config
from .data import DyadicDataset
from .diffusion import DyaDiTSystem
from .fid import MotionFIDCallback
from .visualization import ValidationRenderCallback


class DyadicDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def setup(self, stage=None):
        self.train_dataset = DyadicDataset(self.config)
        validation_root = self.config["paths"].get("val_data_root")
        if not validation_root:
            raise ValueError(
                "Set paths.val_data_root to a separate validation data directory"
            )
        self.val_dataset = DyadicDataset(
            self.config, data_root=validation_root, random_style=False
        )

    def _loader(self, dataset, shuffle):
        settings = self.config["data"]
        workers = settings["num_workers"]
        return DataLoader(
            dataset,
            batch_size=settings["batch_size"],
            shuffle=shuffle,
            num_workers=workers,
            pin_memory=settings["pin_memory"],
            persistent_workers=workers > 0,
        )

    def train_dataloader(self):
        return self._loader(self.train_dataset, True)

    def val_dataloader(self):
        return self._loader(self.val_dataset, False)

    def get_visualization_dataset(self, pose_length):
        """Long inference clips are separate from the 96-frame loss datasets."""
        if pose_length == self.val_dataset.pose_length:
            return self.val_dataset
        return DyadicDataset(
            self.config,
            data_root=self.val_dataset.root,
            pose_length=pose_length,
            random_style=False,
        )


def main():
    parser = argparse.ArgumentParser(
        description="Train the core DyaDiT latent diffusion model"
    )
    parser.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parents[1] / "configs" / "core.yaml"),
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Typed YAML override; repeat as needed",
    )
    checkpoint_group = parser.add_mutually_exclusive_group()
    checkpoint_group.add_argument(
        "--resume", help="Resume a checkpoint produced by this core training entry"
    )
    checkpoint_group.add_argument(
        "--weights",
        help="Initialize weights from an original or core Lightning checkpoint",
    )
    args = parser.parse_args()
    config = load_config(args.config, args.set)
    pl.seed_everything(config["seed"], workers=True)
    output = Path(config["paths"]["output_dir"])
    if int(os.environ.get("RANK", "0")) == 0:
        output.mkdir(parents=True, exist_ok=True)
        with (output / "config.yaml").open("w") as stream:
            yaml.safe_dump(config, stream, sort_keys=False)
    system = DyaDiTSystem(config)
    if args.weights:
        load_system_weights(system, args.weights)
    checkpoint = config["training"]["checkpoint"]
    callbacks = [
        ModelCheckpoint(
            dirpath=output / "checkpoints",
            filename="dyadit-{epoch:04d}-{val/loss:.4f}",
            monitor="val/loss",
            mode="min",
            save_top_k=checkpoint["save_top_k"],
            save_last=True,
            auto_insert_metric_name=False,
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]
    if config.get("fid", {}).get("enabled", False):
        callbacks.append(MotionFIDCallback(config))
    if config.get("visualization", {}).get("enabled", False):
        callbacks.append(ValidationRenderCallback(config))
    logger_settings = config.get("logger", {})
    if logger_settings.get("type", "wandb") == "wandb":
        from pytorch_lightning.loggers import WandbLogger

        logger = WandbLogger(
            project=logger_settings.get("project", "dyadit"),
            entity=logger_settings.get("entity"),
            name=config["exp_name"],
            save_dir=str(output),
        )
    else:
        logger = TensorBoardLogger(str(output), name="logs")
    trainer = pl.Trainer(
        default_root_dir=str(output),
        callbacks=callbacks,
        logger=logger,
        **config["trainer"]
    )
    trainer.fit(system, datamodule=DyadicDataModule(config), ckpt_path=args.resume)


if __name__ == "__main__":
    main()
