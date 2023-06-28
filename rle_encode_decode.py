import numpy as np
import pandas as pd


# ref https://www.kaggle.com/code/paulorzp/run-length-encode-and-decode
def rle_encode(img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''

    if not img.any():
        return np.nan

    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def rle_decode(mask_rle, shape):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background
    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T  # Needed to align to RLE direction


# the masks in "train_ship_segmentations_v2.csv" are sometimes divided into many rows
# combine masks and return numpy array
def masks_to_array(mask_df, shape=(768, 768)):
    """
    Take many rows of encoded pixels and turn them into one mask
    """
    # create zeros mask to fill if needed
    result_mask = np.zeros(shape, dtype=np.uint8)

    # if there is more than one mask we need to get rid of extra dimension
    masks = mask_df.values.flatten()

    # combine masks
    for mask in masks:
        if not pd.isnull(mask):
            result_mask |= rle_decode(mask, shape)

    return result_mask
