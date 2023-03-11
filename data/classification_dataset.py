import glob
import os
import re
import random
from tqdm import tqdm

from torch.utils.data import Dataset
from torchvision.io import read_image

from data import ops


class MNISTImageDataset(Dataset):
    def __init__(self, input_dir_path, classes, steps=1000, image_shape=(256, 256), transform=None):
        self.classes = classes
        self.steps = steps
        self.image_shape = image_shape
        self.transform = transform
        self.image_dict = self.prepare_image_dict(input_dir_path, classes)

    def __len__(self):
        return self.steps

    def __getitem__(self, idx):
        class_index = random.choice(list(range(len(self.classes))))
        class_label = self.classes[class_index]
        image_path = random.choice(self.image_dict[class_label])
        image = read_image(image_path)
        if self.transform:
            image = self.transform(image)
        image, _, _ = ops.resize_and_padding(image, self.image_shape)
        return image, class_index

    @staticmethod
    def prepare_image_dict(input_dir_path, classes):
        image_dict = {}
        for class_label in classes:
            dir_path_list = glob.glob(os.path.join(input_dir_path, f'**/{class_label}/'), recursive=True)
            if len(dir_path_list) > 0:
                dir_path = dir_path_list[0]
            else:
                print(f'not found:{class_label}')
            image_dict[class_label] = []
            image_path_list = [file_path for file_path in glob.glob(os.path.join(dir_path, '**/*.*'), recursive=True) if
                               re.search('.*\.(png|jpg|bmp)$', file_path)]
            for image_path in tqdm(image_path_list, desc=f'_prepare_image_dict:{class_label}'):
                image_dict[class_label].append(image_path)
        return image_dict
