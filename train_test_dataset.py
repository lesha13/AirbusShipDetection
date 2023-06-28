import os
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
from tensorflow.keras.utils import Sequence

from rle_encode_decode import masks_to_array


# define paths to files
PATH = "./"
DATA_PATH = os.path.join(PATH, "dataset")
TRAIN_IMGS = os.path.join(DATA_PATH, "train_v2")
TEST_IMGS = os.path.join(DATA_PATH, "test_v2")
TRAIN_MASKS = os.path.join(DATA_PATH, "train_ship_segmentations_v2.csv")

# other constants
BATCH_SIZE = 32

# get test image ids
test_ids = os.listdir(TEST_IMGS)

# get train image ids and get encoded masks
image_ids = os.listdir(TRAIN_IMGS)
df = pd.read_csv(TRAIN_MASKS)

# take only 20000 empty masks and all masks with ships
empty_masks = df[df.EncodedPixels.isnull()][:20_000:]
masks_with_ships = df[~df.EncodedPixels.isnull()]
encoded_masks = pd.concat([empty_masks, masks_with_ships]).set_index("ImageId")

# ensure the correctness of image ids and shuffle them
image_ids = list(set(image_ids) & set(encoded_masks.index))
np.random.shuffle(image_ids)

# divide train dataset into train and validation datasets
division = .8
i = np.ceil(len(image_ids) * division).astype(int)

train_ids = image_ids[:i:]
validation_ids = image_ids[i+1::]


# create 2 classes based on keras.utils.Sequence
# these objects' purpose is to load data gradually
# it's mandatory to override __len__ and __getitem__ methods when inherit from Sequence


# every iteration returns batch of up to batch_size images from directory
class ImagesSequence(Sequence):
    def __init__(self, image_ids: list, batch_size: int):
        self.images_ids = image_ids
        self.batch_size = batch_size

    def __len__(self):
        return np.ceil(len(self.images_ids) / self.batch_size).astype(int)

    def __getitem__(self, idx):
        slc = slice(idx * self.batch_size, (idx + 1) * self.batch_size)
        X = np.array([
            cv2.imread(os.path.join(TEST_IMGS, img)) for img in self.images_ids[slc]
        ]).astype("float32") / 255.
        return X


# every iteration returns batch of up to batch_size images from directory and masks from pandas dataframe
class ImagesMasksSequence(ImagesSequence):
    def __init__(self, image_ids: list, encoded_masks: pd.DataFrame, batch_size: int):
        super().__init__(image_ids, batch_size)
        self.encoded_masks = encoded_masks

    def __getitem__(self, idx):
        slc = slice(idx * self.batch_size, (idx + 1) * self.batch_size)
        X = np.array([
            cv2.imread(os.path.join(TRAIN_IMGS, img)) for img in self.images_ids[slc]
        ]).astype("float32") / 255.
        y = np.array([
            masks_to_array(self.encoded_masks.loc[mask]) for mask in self.images_ids[slc]
        ]).astype("float32")
        return X, y


# creates and returns train dataset
def get_train_dataset():
    dataset = ImagesMasksSequence(train_ids, encoded_masks, BATCH_SIZE)
    return dataset


# creates and returns validation dataset
def get_validation_dataset():
    dataset = ImagesMasksSequence(validation_ids, encoded_masks, BATCH_SIZE)
    return dataset


# creates and returns test dataset
def get_test_dataset():
    dataset = ImagesSequence(test_ids, BATCH_SIZE)
    return dataset
