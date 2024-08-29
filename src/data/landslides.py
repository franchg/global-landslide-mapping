import glob
import json
import os
import os.path as osp
import warnings
from copy import deepcopy
from typing import List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from sklearn.model_selection import train_test_split
from tifffile import TiffFile
from torch.utils.data import Dataset, IterableDataset
from tqdm import tqdm


class AnchorCrop(torch.nn.Module):
    """Crop the image in a sample using the anchor point as the center. If the anchor point is on
    the edge of the image, the crop will be shifted to the closest valid position (no padding is
    used).

    Args:
        output_size (tuple or int): Desired output size.
        anchor (tuple): Anchor point (x, y) to use as the center of the crop.
    """

    def __init__(self, output_size, anchor):
        assert isinstance(output_size, (int, tuple))
        assert isinstance(anchor, (tuple))
        self.output_size = output_size
        if isinstance(self.output_size, int):
            self.output_size = (output_size, output_size)
        self.anchor = anchor

    def __call__(self, sample: torch.Tensor):
        image = sample
        h, w = image.shape[1:]
        new_h, new_w = self.output_size

        top = max(0, self.anchor[0] - new_h // 2)
        left = max(0, self.anchor[1] - new_w // 2)

        # crop is too big for the image, shift it
        if top + new_h > h:
            top = h - new_h
        if left + new_w > w:
            left = w - new_w

        return image[:, top : top + new_h, left : left + new_w]


class GlobalLandslideData(Dataset):
    NR_CH = 15

    def __init__(
        self,
        input_dir: str,
        label_dir: str,
        random_crop: int = 256,
        random_flip: bool = True,
        normalize: Union[bool, pd.DataFrame] = True,
        in_memory: bool = True,
        channels: Optional[List[int]] = None,
    ):
        self.input_dir = input_dir
        self.label_dir = label_dir
        self.crop = random_crop
        self.channels = channels

        label_files = [f for f in os.listdir(self.label_dir) if f.endswith(".tif")]
        label_files.sort()
        input_files = [f for f in os.listdir(self.input_dir) if f.endswith(".tif")]
        self.files = []
        for lf in label_files:
            prefix = lf.split("_")[0]
            for inptf in input_files:
                if inptf.split("_")[0] == prefix:
                    self.files.append({"input_file": inptf, "label_file": lf})
                    break

        self.transforms = [T.ToTensor()]

        self.flip = (
            None
            if not random_flip
            else [
                T.RandomHorizontalFlip(p=0.5),
                T.RandomVerticalFlip(p=0.5),
                T.RandomApply([T.RandomRotation((90, 90))], p=0.5),
            ]
        )

        # preload images into memory if needed
        # compute band metadata and normalize per band
        self.normalize = None
        self.in_memory = False
        if isinstance(normalize, pd.DataFrame):
            self.metadata = normalize
            self.normalize = T.Normalize(mean=self.metadata.band_mean, std=self.metadata.band_std)
            if in_memory:
                self.load_in_memory()
        elif normalize:
            self.metadata = self.compute_metadata(load_in_memory=in_memory)
            self.normalize = T.Normalize(mean=self.metadata.band_mean, std=self.metadata.band_std)
        elif not normalize and in_memory:
            self.metadata = self.compute_metadata(load_in_memory=in_memory)
        elif not normalize and not in_memory:
            self.metadata = None
        else:
            raise ValueError("Invalid combination of arguments")

        self._len = len(self.files)

    def load_in_memory(self):
        for f in tqdm(self.files, desc="Reading dataset"):
            input_file = osp.join(self.input_dir, f["input_file"])
            with TiffFile(input_file) as input_tif:
                input_arr = input_tif.pages[0].asarray()

            label_file = osp.join(self.label_dir, f["label_file"])
            with TiffFile(label_file) as label_tif:
                label_arr = label_tif.pages[0].asarray()

            f["input_arr"] = input_arr
            f["label_arr"] = label_arr

        self.in_memory = True

    def compute_metadata(self, save_path: Optional[str] = None, load_in_memory: bool = True):
        # per band metadata and statistics
        df = pd.DataFrame()
        df["band_min"] = np.ones(self.NR_CH) * np.nan
        df["band_max"] = np.ones(self.NR_CH) * np.nan
        df["psum"] = np.zeros(self.NR_CH)
        df["psum_sq"] = np.zeros(self.NR_CH)
        df["pvalid"] = np.zeros(self.NR_CH)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", r"All-NaN (slice|axis) encountered")
            # ####### compute statistics per band, excluding nan pixels
            for f in tqdm(self.files, desc="Reading dataset"):
                input_file = osp.join(self.input_dir, f["input_file"])
                with TiffFile(input_file) as input_tif:
                    input_arr = input_tif.pages[0].asarray()
                    input_shape = input_arr.shape
                    assert input_shape[0] >= self.crop
                    assert input_shape[1] >= self.crop
                    assert input_shape[2] == self.NR_CH
                    df.psum += np.nansum(input_arr, axis=(0, 1))
                    df.psum_sq += np.nansum(input_arr**2, axis=(0, 1))
                    df.pvalid += (~np.isnan(input_arr)).sum(axis=(0, 1))
                    minval = np.nanmin(input_arr, axis=(0, 1))
                    maxval = np.nanmax(input_arr, axis=(0, 1))
                    df.band_min = np.nanmin(np.array([df.band_min, minval]), axis=0)
                    df.band_max = np.nanmax(np.array([df.band_max, maxval]), axis=0)

                label_file = osp.join(self.label_dir, f["label_file"])
                with TiffFile(label_file) as label_tif:
                    label_arr = label_tif.pages[0].asarray()
                    label_shape = label_arr.shape
                    assert len(label_shape) == 2
                    assert label_shape[0] == input_shape[0]
                    assert label_shape[1] == input_shape[1]
                    assert np.all(np.unique(label_arr) == np.array([0, 1]))

                if load_in_memory:
                    f["input_arr"] = input_arr
                    f["label_arr"] = label_arr

        # mean and std
        df["band_mean"] = df.psum / df.pvalid
        df["band_var"] = (df.psum_sq / df.pvalid) - (df.band_mean**2)
        df["band_std"] = np.sqrt(df.band_var)
        df.pvalid = df.pvalid.astype(int)
        assert np.all(df.band_min < df.band_max)

        self.in_memory = load_in_memory

        if save_path is not None:
            df.to_csv(save_path, index=False, header=True)
        return df

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, idx: int):
        f = self.files[idx]
        if self.in_memory:
            inpt = f["input_arr"]
            label = f["label_arr"]
        else:
            inpt = TiffFile(osp.join(self.input_dir, f["input_file"])).pages[0].asarray()
            label = TiffFile(osp.join(self.label_dir, f["label_file"])).pages[0].asarray()

        # stack input and label to apply transforms
        # label is the last channel
        inpt = np.concatenate([inpt, label[..., None]], axis=2)

        # apply transforms
        transforms = self.transforms.copy()
        if self.crop > 0:
            # compute rando anchor coordinates from label that contains landslide
            anchors = np.stack(np.where(label == 1))
            if len(anchors[0]) == 0:
                # no landslide in this image
                # choose a random anchor
                anchors = np.array(
                    [
                        np.random.randint(0, label.shape[0], 1),
                        np.random.randint(0, label.shape[1], 1),
                    ]
                )
            random_pick = int(np.random.randint(0, anchors.shape[1], 1))
            anchor = tuple(anchors[:, random_pick])
            transforms.append(AnchorCrop(self.crop, anchor))
        else:
            # get height and width
            h, w = inpt.shape[:2]
            # crop the image to be a multiple of 32
            h = h - (h % 32)
            w = w - (w % 32)
            transforms.append(T.CenterCrop((h, w)))
        if self.flip is not None:
            transforms.extend(self.flip)
        transforms = T.Compose(transforms)
        inpt = transforms(inpt)

        # split input and label
        # label is the last channel
        label = inpt[-1:, ...]
        inpt = inpt[:-1, ...]

        # normalize input
        # label is not normalized
        if self.normalize is not None:
            inpt = self.normalize(inpt)
            # fill nan with 0
            inpt = torch.nan_to_num(inpt, nan=0.0)

        # filter input channels if needed
        if self.channels is not None:
            inpt = inpt[self.channels, ...]

        return inpt, label

    # split the dataset into train and validation by copying the dataset and changing the files and length
    # returns a tuple of 2 datasets
    def split_dataset(self, split: float = 0.8):
        assert split > 0 and split < 1
        # copy the dataset
        train_ds = deepcopy(self)
        val_ds = deepcopy(self)
        # split the file lis randomly
        train_files, val_files = train_test_split(self.files, train_size=split)
        # update the files
        train_ds.files = train_files
        val_ds.files = val_files
        # update the length
        train_ds._len = len(train_ds.files)
        val_ds._len = len(val_ds.files)
        return train_ds, val_ds

    def plot(self, idx: int, figsize: tuple = (12, 12)):
        inpt, label = self.__getitem__(idx)
        fig, axs = plt.subplots(4, 4, figsize=figsize)
        for i in range(15):
            axs[i // 4, i % 4].imshow(inpt[i])
            axs[i // 4, i % 4].set_title(
                f"Band {i}, mean: {inpt[i].mean():.2f}, std: {inpt[i].std():.2f}"
            )
        # plot the label
        axs[3, 3].imshow(label.squeeze())
        axs[3, 3].set_title("GROUND TRUTH")
        plt.tight_layout()
        plt.show()

    def infinite_iterable(self, shuffle: bool = True):
        while True:
            indices = np.arange(len(self))
            if shuffle:
                np.random.shuffle(indices)
            for idx in indices:
                yield self.__getitem__(idx)


class LandslideData(Dataset):
    def __init__(self, input_layers_dir: str, ground_truth_dir: str, transforms=None):
        """
        :param input_layers_dir: folder where all the input layers data is saved.
                                 There must be one sub-folder for each layer with the numbered files representing
                                 each one of the mapped areas. Only tiff files are supported. For example:

                                 input_layers_dir/
                                     |
                                     +-- dem/
                                     |    |
                                     |    +-- dem_01.tif (540 x 340 size) -----------+
                                     |    +-- dem_02.tif (364 x 480 size)            |
                                     |    +-- layer_config.json                      |
                                     |                                               |
                                     +-- aspect/                                     |
                                     |    |                                          |
                                     |    +-- aspect_01.tif (540 x 340 size) --------+--->> width and height must match
                                     |    +-- aspect_02.tif (364 x 480 size)         |
                                     |    +-- layer_config.json                      |
                                     |                                               |
                                     +-- sentinel/                                   |
                                         |                                           |
                                         +-- sentinel_01.tif (540 x 340 size) -------+
                                         +-- sentinel_02.tif (364 x 480 size)
                                         +-- layer_config.json

        :param ground_truth_dir: folder where all the ground truth data is saved. Example:

                                 ground_truth_dir/
                                     |
                                     +-- gt_01.tif (540 x 340 size)
                                     +-- gt_02.tif (364 x 480 size)
                                     +-- layer_config.json

        The layer_config.json file specify the min and max value for each layer used for data minmax normalization.
        """

        self.transforms = transforms

        # collect all the input folders (input layers)
        self.input_layers_dir = input_layers_dir
        self.input_layer_list = os.listdir(self.input_layers_dir)
        self.input_layer_list.sort()

        # collect all the ground truth files
        self.ground_truth_dir = ground_truth_dir
        self.ground_truth_list = os.listdir(self.ground_truth_dir)
        self.ground_truth_list.sort()

        # create the ground truth list by reading all the tiff files
        # prepare the input_areas list with empty data ready to be filled
        self.gt = list()
        self.input_areas = list()
        for ground_truth_file in self.ground_truth_list:
            gt_full_path = osp.join(self.ground_truth_dir, ground_truth_file)
            with TiffFile(gt_full_path) as tif:
                self.gt.append(
                    torch.tensor(tif.pages[0].asarray()[np.newaxis, ...], dtype=torch.float32)
                )
                self.input_areas.append(list())

        # cycle through all the input layers
        for layer in self.input_layer_list:

            # read layer config (min max values) and file list (list length must match the ground truth).
            layer_config = json.load(
                open(osp.join(self.input_layers_dir, layer, "layer_config.json"), "rb")
            )
            input_files = glob.glob(osp.join(self.input_layers_dir, layer) + "/*.tif")
            input_files.sort()
            assert len(input_files) == len(self.ground_truth_list)

            # add to each input area the layer data by concatenating it as a channel
            # normalization is applied before concatenation
            for i, input_file in enumerate(input_files):
                with TiffFile(input_file) as tif:
                    arr = tif.pages[0].asarray().astype(np.float)
                    arr = (arr - layer_config["min"]) / (layer_config["max"] - layer_config["min"])
                    # print(layer, input_file, arr.shape[:2], self.gt[i].shape)
                    assert arr.shape[:2] == self.gt[i].shape[1:]
                    if len(arr.shape) == 2:
                        self.input_areas[i].append(arr)
                    elif len(arr.shape) == 3:
                        for j in range(arr.shape[2]):
                            self.input_areas[i].append(arr[..., j])
                    else:
                        raise NotImplementedError()

        # cast the list of channels into a torch tensor
        for i, input_area in enumerate(self.input_areas):
            self.input_areas[i] = torch.tensor(np.array(input_area), dtype=torch.float32)

    def __len__(self):
        return len(self.gt)

    def __getitem__(self, index):
        inputs = self.input_areas[index]
        gt_mask = self.gt[index]

        # check to see if we are applying any transformations
        if self.transforms is not None:
            # apply the transformations to both input layers and its ground truth
            inputs = self.transforms(inputs)
            gt_mask = self.transforms(gt_mask)

        return inputs, gt_mask


class InfiniteGlobalLandslideData(Dataset):
    def __init__(
        self,
        dataset: GlobalLandslideData,
        stop_after: int,
        shuffle: bool = True,
    ):
        self.dataset = dataset
        self.stop_after = stop_after
        self.iter = iter(self.dataset.infinite_iterable(shuffle))

    def __getitem__(self, index):
        if index >= self.stop_after:
            raise IndexError()
        return next(self.iter)

    def __len__(self):
        return self.stop_after
