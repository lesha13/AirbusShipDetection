import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model

from train_test_dataset import get_test_dataset, BATCH_SIZE
from dice_coefficient import dice_metric, dice_loss
from rle_encode_decode import rle_encode

# define paths to files
PATH = "./"
SUBMISSION = os.path.join(PATH, "dataset/sample_submission_v2.csv")
MODEL_PATH = os.path.join(PATH, "model.h5")
RESULTS = os.path.join(PATH, "results.csv")


def inference():
    # get test dataset
    test_dataset = get_test_dataset()

    # load model and specify non-keras objects
    model = load_model(
        MODEL_PATH,
        custom_objects={
            "dice_loss": dice_loss,
            "dice_metric": dice_metric,
        }
    )

    # read sample submission
    submission = pd.read_csv(SUBMISSION)

    for i, batch in enumerate(test_dataset):
        # predict values for batch and cast them to int
        y_hat = model(batch)
        y_hat = np.round(y_hat).astype(int)

        # insert encoded masks to certain positions
        slc = slice(i * BATCH_SIZE, (i + 1) * BATCH_SIZE)
        submission.EncodedPixels[slc] = [rle_encode(img) for img in y_hat]
    else:
        # save results to csv file
        submission.set_index("ImageId").to_csv(RESULTS)
