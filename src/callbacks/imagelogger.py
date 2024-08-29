import os

import numpy as np
import pytorch_lightning as pl
import torch
import torchvision
from PIL import Image
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities import rank_zero_only


class ImageLogger(Callback):
    def __init__(
        self,
        max_images: int,
        train_batch_frequency: int = 0,
        val_batch_frequency: int = 0,
        test_batch_frequency: int = 0,
    ):
        super().__init__()
        self.max_images = max_images
        self.train_batch_frequency = train_batch_frequency
        self.val_batch_frequency = val_batch_frequency
        self.test_batch_frequency = test_batch_frequency
        self.tb_logger = None
        self.global_step = 0

    def setup(self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str) -> None:
        for logger in trainer.loggers:
            if isinstance(logger, pl.loggers.TensorBoardLogger):
                self.tb_logger = logger.experiment
                break

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs,
        batch,
        batch_idx,
        dataloader_idx=None,
    ):
        if self.train_batch_frequency > 0 and batch_idx % self.train_batch_frequency == 0:
            self.log_images(pl_module, outputs["targets"], outputs["preds"], "train", batch_idx)

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs,
        batch,
        batch_idx,
        dataloader_idx=None,
    ):
        if self.val_batch_frequency > 0 and batch_idx % self.val_batch_frequency == 0:
            self.log_images(pl_module, outputs["targets"], outputs["preds"], "val", batch_idx)

    def on_test_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs,
        batch,
        batch_idx,
        dataloader_idx=None,
    ):
        if self.test_batch_frequency > 0 and batch_idx % self.test_batch_frequency == 0:
            self.log_images(pl_module, outputs["targets"], outputs["preds"], "test", batch_idx)

    @rank_zero_only
    def log_images(self, pl_module, label, outputs, split, batch_idx):
        # log to tensorboard logger if available
        if self.tb_logger is not None:
            is_train = pl_module.training
            if is_train:
                pl_module.eval()

            with torch.no_grad():
                # limit to max_images
                N = min(outputs.shape[0], self.max_images)
                # log both outputs and ground truth from batch
                otp = outputs[:N].round().type(torch.int8)
                lbl = label[:N].round().type(torch.int8)
                images = torch.cat([otp, lbl], dim=0)
                grid = torchvision.utils.make_grid(images, nrow=4, pad_value=1)
                self.tb_logger.add_image(split, grid, global_step=self.global_step)

                image_rg = torch.cat([otp, lbl], dim=1)
                # add a channel of zeros on dim 1 to create a 3 channel image
                image_rgb = torch.cat([torch.zeros_like(image_rg[:, :1, :, :]), image_rg], dim=1)
                self.tb_logger.add_images(f"{split}/diff", image_rgb, global_step=self.global_step)
                self.global_step += 1

            if is_train:
                pl_module.train()

    # @rank_zero_only
    # def _testtube(self, pl_module, images, batch_idx, split):
    #     for k in images:
    #         grid = torchvision.utils.make_grid(images[k])
    #         grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w

    #         tag = f"{split}/{k}"
    #         pl_module.logger.experiment.add_image(tag, grid, global_step=pl_module.global_step)

    # @rank_zero_only
    # def log_local(self, save_dir, split, images, global_step, current_epoch, batch_idx):
    #     root = os.path.join(save_dir, "images", split)
    #     for k in images:
    #         grid = torchvision.utils.make_grid(images[k], nrow=4)
    #         if self.rescale:
    #             grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
    #         grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
    #         grid = grid.numpy()
    #         grid = (grid * 255).astype(np.uint8)
    #         filename = "{}_gs-{:06}_e-{:06}_b-{:06}.png".format(
    #             k, global_step, current_epoch, batch_idx
    #         )
    #         path = os.path.join(root, filename)
    #         os.makedirs(os.path.split(path)[0], exist_ok=True)
    #         Image.fromarray(grid).save(path)

    # def log_img(self, pl_module, batch, batch_idx, split="train"):
    #     check_idx = batch_idx if self.log_on_batch_idx else pl_module.global_step
    #     if (
    #         self.check_frequency(check_idx)
    #         and hasattr(pl_module, "log_images")  # batch_idx % self.batch_freq == 0
    #         and callable(pl_module.log_images)
    #         and self.max_images > 0
    #     ):
    #         logger = type(pl_module.logger)

    #         is_train = pl_module.training
    #         if is_train:
    #             pl_module.eval()

    #         with torch.no_grad():
    #             images = pl_module.log_images(batch, split=split, **self.log_images_kwargs)

    #         for k in images:
    #             N = min(images[k].shape[0], self.max_images)
    #             images[k] = images[k][:N]
    #             if isinstance(images[k], torch.Tensor):
    #                 images[k] = images[k].detach().cpu()
    #                 if self.clamp:
    #                     images[k] = torch.clamp(images[k], -1.0, 1.0)

    #         self.log_local(
    #             pl_module.logger.save_dir,
    #             split,
    #             images,
    #             pl_module.global_step,
    #             pl_module.current_epoch,
    #             batch_idx,
    #         )

    #         logger_log_images = self.logger_log_images.get(logger, lambda *args, **kwargs: None)
    #         logger_log_images(pl_module, images, pl_module.global_step, split)

    #         if is_train:
    #             pl_module.train()

    # def check_frequency(self, check_idx):
    #     if ((check_idx % self.batch_freq) == 0 or (check_idx in self.log_steps)) and (
    #         check_idx > 0 or self.log_first_step
    #     ):
    #         try:
    #             self.log_steps.pop(0)
    #         except IndexError as e:
    #             print(e)
    #             pass
    #         return True
    #     return False

    # def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
    #     if not self.disabled and (pl_module.global_step > 0 or self.log_first_step):
    #         self.log_img(pl_module, batch, batch_idx, split="train")

    # def on_validation_batch_end(
    #     self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    # ):
    #     if not self.disabled and pl_module.global_step > 0:
    #         self.log_img(pl_module, batch, batch_idx, split="val")
    #     if hasattr(pl_module, "calibrate_grad_norm"):
    #         if (pl_module.calibrate_grad_norm and batch_idx % 25 == 0) and batch_idx > 0:
    #             self.log_gradients(trainer, pl_module, batch_idx=batch_idx)

    # def validation_step(self, batch: Any, batch_idx: int):

    #     imgs, y_true = batch
    #     y_pred = self(imgs)
    #     val_loss = self.nn_criterion(y_pred, y_true)
    #     self.log("val_loss", val_loss)

    #     if batch_idx % 10: # Log every 10 batches
    #         self.log_tb_images((imgs, y_true, y_pred, batch_idx))

    #     return loss

    # def log_tb_images(self, viz_batch) -> None:

    #      # Get tensorboard logger
    #      tb_logger = None
    #      for logger in self.trainer.loggers:
    #         if isinstance(logger, pl_loggers.TensorBoardLogger):
    #             tb_logger = logger.experiment
    #             break

    #     if tb_logger is None:
    #             raise ValueError('TensorBoard Logger not found')

    #     # Log the images (Give them different names)
    #     for img_idx, (image, y_true, y_pred, batch_idx) in enumerate(zip(*viz_batch)):
    #         tb_logger.add_image(f"Image/{batch_idx}_{img_idx}", image, 0)
    #         tb_logger.add_image(f"GroundTruth/{batch_idx}_{img_idx}", y_true, 0)
    #         tb_logger.add_image(f"Prediction/{batch_idx}_{img_idx}", y_pred, 0)
