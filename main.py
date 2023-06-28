import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2

from rle_encode_decode import masks_to_array
from model_training import train
from model_tuning import tune
from model_inference import inference


def main():
    # train()
    # tune()
    # inference()

    # plot random image and predicted mask
    TEST_IMGS = os.path.join("./dataset", "test_v2")
    df = pd.read_csv("results.csv")

    image_id = np.random.choice(os.listdir(TEST_IMGS))
    image = cv2.imread(os.path.join(TEST_IMGS, image_id))
    mask = masks_to_array(df.set_index("ImageId").loc[image_id])

    fig, ax = plt.subplots(1, 2)

    ax[0].imshow(image)
    ax[0].set_xlabel("Image")
    ax[1].imshow(mask, cmap="gray")
    ax[1].set_xlabel("Mask")

    plt.show()


if __name__ == "__main__":
    main()
