from typing import Any, Dict, List, Optional

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split

from src.data.landslides import GlobalLandslideData, InfiniteGlobalLandslideData


class GlobalLandslideDataModule(LightningDataModule):
    def __init__(
        self,
        train_input_dir: str = "data/landslides/train/input/",
        train_label_dir: str = "data/lanslides/train/label/",
        test_input_dir: str = "data/lanslides/test/input/",
        test_label_dir: str = "data/lanslides/test/label/",
        batch_size: int = 4,
        num_workers: int = 0,
        pin_memory: bool = False,
        random_crop: int = 256,
        random_flip: bool = False,
        normalize: bool = True,
        in_memory: bool = True,
        train_val_split: float = 0.8,
        samples_per_epoch: int = None,
        channels: Optional[List[int]] = None,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # self.pin_memory = pin_memory
        # self.num_workers = num_workers
        # self.batch_size = batch_size
        # self.train_dataset_kwargs = train_dataset_kwargs
        # self.train_input_dir = train_input_dir
        # self.train_label_dir = train_label_dir
        # self.test_input_dir = test_input_dir
        # self.test_label_dir = test_label_dir

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    def prepare_data(self):
        # download, split, etc...
        # only called on 1 GPU/TPU in distributed
        pass

    def setup(self, stage: Optional[str] = None):
        # make assignments here (val/train/test split)
        # called on every process in DDP
        if self.data_train is None and self.data_val is None:
            train_val = GlobalLandslideData(
                input_dir=self.hparams.train_input_dir,
                label_dir=self.hparams.train_label_dir,
                random_crop=self.hparams.random_crop,
                random_flip=self.hparams.random_flip,
                normalize=self.hparams.normalize,
                in_memory=self.hparams.in_memory,
                channels=self.hparams.channels,
            )
            self.data_train, self.data_val = train_val.split_dataset(self.hparams.train_val_split)
            if self.hparams.samples_per_epoch is not None:
                self.data_train = InfiniteGlobalLandslideData(
                    self.data_train,
                    self.hparams.samples_per_epoch,
                    shuffle=True,
                )
            self.data_val.crop = 0
            self.data_val.flip = None
            # self.data_train, self.data_val = random_split(train_val, [0.8, 0.2])
            self.data_test = GlobalLandslideData(
                input_dir=self.hparams.test_input_dir,
                label_dir=self.hparams.test_label_dir,
                random_crop=0,
                random_flip=False,
                normalize=train_val.metadata,
                in_memory=self.hparams.in_memory,
                channels=self.hparams.channels,
            )

    def train_dataloader(self):
        return DataLoader(
            self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.data_val,
            batch_size=1,
            num_workers=1,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.data_test,
            batch_size=1,
            num_workers=1,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass
