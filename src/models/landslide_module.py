from typing import Any, List, Optional, Union

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from pytorch_lightning import LightningModule
from tensorboard.backend.event_processing import event_accumulator
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification import F1Score
from torchmetrics.classification.accuracy import Accuracy
from tqdm import tqdm


class LandslideLitModule(LightningModule):
    """LightningModule for Landslide classification.

    A LightningModule organizes your PyTorch code into 6 sections:
        - Computations (init)
        - Train loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Prediction Loop (predict_step)
        - Optimizers and LR Schedulers (configure_optimizers)

    Docs:
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
    """

    def __init__(
        self,
        net: torch.nn.Module,
        loss: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.net = net
        self.loss = loss

        # loss function
        # self.criterion = loss
        # self.criterion = torch.nn.BCEWithLogitsLoss()

        # metric objects for calculating and averaging accuracy across batches
        self.train_acc = Accuracy(task="binary")
        self.val_acc = Accuracy(task="binary")
        self.test_acc = Accuracy(task="binary")

        # metric objects for calculating and averaging f1 across batches
        self.train_f1 = F1Score(task="binary")
        self.val_f1 = F1Score(task="binary")
        self.test_f1 = F1Score(task="binary")

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_acc_best = MaxMetric()
        self.val_f1_best = MaxMetric()

    def forward(self, x: torch.Tensor):
        return self.net(x)

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so we need to make sure val_acc_best doesn't store accuracy from these checks
        self.val_acc_best.reset()

    def model_step(self, batch: Any):
        x, y = batch
        logits = self.forward(x)
        # loss = self.criterion(logits, y)
        loss = self.loss(logits, y)
        # preds = torch.argmax(logits, dim=1)
        preds = torch.sigmoid(logits)
        return loss, preds, y

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.train_loss(loss)
        self.train_acc(preds, targets)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)

        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()` below
        # remember to always return loss from `training_step()` or backpropagation will fail!
        return {"loss": loss, "preds": preds, "targets": targets}

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`

        # Warning: when overriding `training_epoch_end()`, lightning accumulates outputs from all batches of the epoch
        # this may not be an issue when training on mnist
        # but on larger datasets/models it's easy to run into out-of-memory errors

        # consider detaching tensors before returning them from `training_step()`
        # or using `on_train_epoch_end()` instead which doesn't accumulate outputs

        pass

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.val_loss(loss)
        self.val_acc(preds, targets)
        self.val_f1(preds, targets)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/f1", self.val_f1, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def validation_epoch_end(self, outputs: List[Any]):
        acc = self.val_acc.compute()  # get current val acc
        f1 = self.val_f1.compute()  # get current val f1
        self.val_acc_best(acc)  # update best so far val acc
        self.val_f1_best(f1)  # update best so far val acc
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/acc_best", self.val_acc_best.compute(), prog_bar=True)
        self.log("val/f1_best", self.val_f1_best.compute(), prog_bar=True)

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.test_loss(loss)
        self.test_acc(preds, targets)
        self.test_f1(preds, targets)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/f1", self.test_f1, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def test_epoch_end(self, outputs: List[Any]):
        pass

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


# load tensorboard data from protobuf folder
def load_tensorboard_scalars(
    path: str, variables: Optional[List[str]] = None, reduce: Optional[str] = None
) -> Union[pd.DataFrame, float]:
    # collect all protobuf files
    import glob

    files = glob.glob(path + "/**/events.out.tfevents.*", recursive=True)
    files = sorted(files, key=lambda x: int(x.split(".")[-1]))
    assert len(files) >= 1
    # cycle through all files and find all variables
    data = {}
    for file in files:
        ea = event_accumulator.EventAccumulator(file)
        ea.Reload()
        tags = ea.Tags()["scalars"]
        if variables is not None:
            tags = [tag for tag in tags if any([v in tag for v in variables])]
        data.update({tag: ea.Scalars(tag) for tag in tags})
    if reduce is not None:
        if reduce == "mean":
            data = {k: np.mean([x.value for x in data[k]]) for k in data}
        elif reduce == "max":
            data = {k: np.max([x.value for x in data[k]]) for k in data}
        elif reduce == "min":
            data = {k: np.min([x.value for x in data[k]]) for k in data}
        elif reduce == "last":
            data = {k: data[k][-1].value for k in data}
        elif reduce == "first":
            data = {k: data[k][0].value for k in data}
        else:
            raise NotImplementedError
        # return scalar if only one variable is requested
        if len(data) == 1:
            return list(data.values())[0]
    if reduce is not None:
        df = pd.DataFrame(data, index=[0])
    else:
        df = pd.DataFrame({tag: [x.value for x in data[tag]] for tag in data})
        df["step"] = [x.step for x in data[list(data.keys())[0]]]
    return df


class LandslideEnsemble:
    def __init__(self, models: Union[List[LandslideLitModule], List[str]]) -> None:
        assert len(models) > 0
        self.models = (
            models
            if models[0] is not str
            else [LandslideLitModule.load_from_checkpoint(m) for m in models]
        )
        # set model to eval mode
        for m in self.models:
            m.eval()

    def __len__(self) -> int:
        return len(self.models)
    
    def __getitem__(self, idx: Union[int, slice]) -> Union[LandslideLitModule, List[LandslideLitModule]]:
        return self.models[idx]

    def __call__(self, x: torch.Tensor, mean: bool = True, top_k: int = None) -> torch.Tensor:
        x_device = x.device
        x = x.to(self.models[0].device)
        with torch.no_grad():
            if top_k is None:
                for model in tqdm(self.models, desc="Ensembling"):
                    res.append(torch.sigmoid(model(x)))
            else:
                res = []
                for i in tqdm(range(top_k), desc="Ensembling"):
                    res.append(torch.sigmoid(self.models[i](x)))
            res = torch.stack(res)
            if mean:
                return res.mean(dim=0).to(x_device)
            return res.to(x_device)
        
    def to(self, device: Union[str, torch.device]) -> "LandslideEnsemble":
        for m in self.models:
            m.to(device)
        return self

    def to_rgb_diff(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        otp = x.round().type(torch.int8)
        lbl = y.round().type(torch.int8)
        image_rg = torch.cat([otp, lbl], dim=1)
        # add a channel of zeros on dim 1 to create a 3 channel image
        image_rgb = torch.cat([torch.zeros_like(image_rg[:, :1, :, :]), image_rg], dim=1)

    def __getitem__(self, idx: int) -> LandslideLitModule:
        return self.models[idx]

    @staticmethod
    # model folders are saved with the following structure, an there are multiple of them (from 0 to N)
    # 0
    # ├── checkpoints
    # │   ├── epoch_010.ckpt
    # │   └── last.ckpt
    # ├── config_tree.log
    # ├── tags.log
    # └── tensorboard
    #     └── version_0
    #         ├── events.out.tfevents.1689845082.airflow.1022171.0
    #         ├── events.out.tfevents.1689845843.airflow.1022171.1
    #         └── hparams.yaml
    def load_best_models_from_multirun(
        path: str, n: int = 10, variable_name: str = "val/f1_best"
    ) -> List[LandslideLitModule]:
        import glob
        import os

        # collect all model folder names
        folders = glob.glob(path + "/**/checkpoints", recursive=True)
        folders = sorted(folders, key=lambda x: int(os.path.split(os.path.split(x)[0])[1]))
        assert len(folders) >= 1

        # cycle through all folders and find the max value of the variable_name
        models = {}
        for folder in tqdm(
            folders, desc=f"Scanning tensorboard logs to find best {variable_name}"
        ):
            models[folder] = load_tensorboard_scalars(
                os.path.join(folder, "../tensorboard"), variables=[variable_name], reduce="max"
            )

        # filter out the best n models
        models = sorted(models.items(), key=lambda x: x[1], reverse=True)[:n]

        # load the models (find the file that matches epoch_XXX.ckpt, which is the best model)
        # use tqdm to show progress bar
        best_models = [
            LandslideLitModule.load_from_checkpoint(
                os.path.join(x[0], [f for f in os.listdir(x[0]) if "epoch_" in f][0])
            )
            for x in tqdm(models, desc=f"Loading {n} models")
        ]

        # add the model name and score to the hparams
        for i, model in enumerate(best_models):
            model.hparams.model_number = os.path.split(os.path.split(models[i][0])[0])[1]
            model.hparams.model_score = models[i][1]
            # save tensorboard hparams.yaml to model hparams
            model.hparams.cfg = OmegaConf.load(
                os.path.join(models[i][0], "../tensorboard/version_0/hparams.yaml")
            )

        return best_models

    def ensemble_summary(self) -> pd.DataFrame:
        # build model summary (model name, score, loss function, learning rate, network architecture)
        # as a pandas dataframe
        summary = pd.DataFrame(
            [
                [
                    model.hparams.model_number,
                    model.hparams.model_score,
                    model.hparams.loss.__class__.__name__,
                    model.hparams.optimizer.keywords["lr"],
                    model.net.__class__.__name__,
                ]
                for model in self.models
            ],
            columns=["model_number", "model_score", "loss", "lr", "net"],
        )
        return summary


if __name__ == "__main__":
    _ = LandslideLitModule(None, None, None, None)
