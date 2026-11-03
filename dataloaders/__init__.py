import itertools
import h5py
import numpy as np
from torch.utils.data import Dataset, Sampler
import os

def iterate_once(iterable):
    """Iterate through the iterable once, returning a shuffled version."""
    return np.random.permutation(iterable)

def iterate_eternally(indices):
    """Iterate through the indices infinitely, yielding shuffled batches."""
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())

def grouper(iterable, n):
    """Collect data into fixed-length chunks or blocks."""
    args = [iter(iterable)] * n
    return zip(*args)

class BaseDataSets(Dataset):
    """Base dataset class for loading medical imaging data."""

    def __init__(
        self,
        base_dir=None,
        split="train",
        num=None,
        transform=None,
    ):
        self._base_dir = base_dir
        self.sample_list = []
        self.split = split
        self.transform = transform

        if self.split == "train":
            with open(self._base_dir + "/train_slices.list", "r") as f:
                self.sample_list = f.readlines()
            self.sample_list = [item.replace("\n", "") for item in self.sample_list]
            if num is not None:
                self.sample_list = self.sample_list[:num]
            print("Train total {} samples".format(len(self.sample_list)))
        elif self.split in ("val", "test"):
            with open(self._base_dir + f"/{self.split}.list", "r") as f:
                self.sample_list = f.readlines()
            self.sample_list = [item.replace("\n", "") for item in self.sample_list]
            print(f"{self.split.capitalize()} total {len(self.sample_list)} samples")

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        is_train = (self.split == "train")
        is_synapse = "synapse" in self._base_dir.lower()

        # 1. 确定文件夹路径（优化了路径拼接，避免混用斜杠）
        if is_train:
            dir_path = os.path.join(self._base_dir, "data", "slices")
        else:
            dir_path = os.path.join(self._base_dir, "data")

        # 2. 动态探测文件名（完美兼容 .h5 和 .npy.h5）
        if case.endswith(".h5"):
            file_name = case
        else:
            test_name_h5 = f"{case}.h5"
            test_name_npy = f"{case}.npy.h5"

            if os.path.exists(os.path.join(dir_path, test_name_h5)):
                file_name = test_name_h5
            elif os.path.exists(os.path.join(dir_path, test_name_npy)):
                file_name = test_name_npy
            else:
                # 如果都找不到，默认用 .h5，让报错指出具体路径
                file_name = test_name_h5

        h5_path = os.path.join(dir_path, file_name)

        # 3. 读取文件（with 自动管理句柄，防止内存泄漏）
        with h5py.File(h5_path, 'r') as h5f:
            image = h5f["image"][:]
            label = h5f["label"][:]

        # 4. Synapse 标签映射
        if is_synapse:
            label[label == 5] = 0
            label[label == 9] = 0
            label[label == 10] = 0
            label[label == 12] = 0
            label[label == 13] = 0
            label[label == 11] = 5

        assert np.all((label >= 0) & (label < 9)), \
            f"标签越界: {np.unique(label)}"

        # 5. 构建样本字典
        sample = {"image": image, "label": label}
        if is_train:
            if self.transform is not None:
                sample = self.transform(sample)

        sample["idx"] = idx
        # 6. 清理 casename 后缀，确保名字干净（用于后续保存预测结果）
        sample["casename"] = case.replace(".npy.h5", "").replace(".h5", "")

        return sample

class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """

    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch for (primary_batch, secondary_batch) in
            zip(grouper(primary_iter, self.primary_batch_size),
                grouper(secondary_iter, self.secondary_batch_size)))

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size


def map_synapse_labels(label):
    label = np.asarray(label).copy()

    label[label == 5] = 0
    label[label == 9] = 0
    label[label == 10] = 0
    label[label == 12] = 0
    label[label == 13] = 0
    label[label == 11] = 5

    return label