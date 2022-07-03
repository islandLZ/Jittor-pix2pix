import glob
import random
import os
import numpy as np

from jittor.dataset.dataset import Dataset
import jittor.transform as transform
from PIL import Image

class ImageDataset(Dataset):
    def __init__(self, root, mode="train", transforms=None):
        super().__init__()
        self.transforms = transform.Compose(transforms)
        self.mode = mode
        #if self.mode == 'train':
        self.files = sorted(glob.glob(os.path.join(root, mode, "imgs") + "/*.*"))# 彩色图片
        self.labels = sorted(glob.glob(os.path.join(root, mode, "labels") + "/*.*"))# 标签（黑色图片）
        self.set_attrs(total_len=len(self.labels))
        print(f"from {mode} split load {self.total_len} images.")

    def __getitem__(self, index):
        label_path = self.labels[index % len(self.labels)]
        photo_id = label_path.split('/')[-1][:-4]
        img_B = Image.open(label_path)
        img_B = Image.fromarray(np.array(img_B).astype("uint8")[:, :, np.newaxis].repeat(3,2))

        
        img_A = Image.open(self.files[index % len(self.files)])
        if np.random.random() < 0.5:
            img_A = Image.fromarray(np.array(img_A)[:, ::-1, :], "RGB")
            img_B = Image.fromarray(np.array(img_B)[:, ::-1, :], "RGB")
        img_A = self.transforms(img_A)
        img_B = self.transforms(img_B)

        # 彩色图片，标签（黑色图片），图片名
        return img_A, img_B, photo_id



'''
class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, mode="train"):
        super().__init__()
        self.transform = transform.Compose(transforms_)
        self.files = sorted(glob.glob(os.path.join(root, mode) + "/*.*"))
        self.set_attrs(total_len=len(self.files))

    def __getitem__(self, index):

        img = Image.open(self.files[index % len(self.files)])
        w, h = img.size
        img_A = img.crop((0, 0, w / 2, h))
        img_B = img.crop((w / 2, 0, w, h))

        if np.random.random() < 0.5:
            img_A = Image.fromarray(np.array(img_A)[:, ::-1, :], "RGB")
            img_B = Image.fromarray(np.array(img_B)[:, ::-1, :], "RGB")

        img_A = self.transform(img_A)
        img_B = self.transform(img_B)

        return img_A, img_B
'''