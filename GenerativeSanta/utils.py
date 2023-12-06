import pathlib

import pandas as pd
import torch
from PIL import Image


class SantaDataset(torch.utils.data.Dataset):
    def __init__(self, txt_file: str = "data/training.txt", transform=None):
        """
        A custom dataset class for loading and applying transforms to the images samples
        from the Santa Claus dataset-

        :param txt_file: Path to the txt file with annotations (image filename, encoding of class).
        :type txt_file: str
        :param transform: callable object, optional transform to be applied to the data
        """
        df = pd.read_csv(txt_file, sep=",")
        self.img_filename = df["filename"].tolist()
        self.img_label = df["is_santa"].tolist()
        self.transform = transform

    def __len__(self):
        return len(self.img_filename)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = pathlib.Path(self.img_filename[idx])
        image = Image.open(img_name)
        image = image.convert("RGB")
        label = self.img_label[idx]

        sample = {"image": image, "label": label}

        if self.transform:
            sample["image"] = self.transform(sample["image"])

        return sample
