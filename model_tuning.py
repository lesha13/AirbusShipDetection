import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint

from train_test_dataset import get_train_dataset, get_validation_dataset
from dice_coefficient import dice_metric, dice_loss

# define paths to files
PATH = "./"
MODEL_PATH = os.path.join(PATH, "model.h5")


def tune():
    # get train and validation dataset
    train_dataset = get_train_dataset()
    validation_dataset = get_validation_dataset()

    # load model and specify non-keras objects
    model = load_model(
        MODEL_PATH,
        custom_objects={
            "dice_loss": dice_loss,
            "dice_metric": dice_metric,
        }
    )

    # train again for 1 epoch
    model.fit(
        train_dataset,
        validation_data=validation_dataset,
        callbacks=[
            ModelCheckpoint(MODEL_PATH, monitor="dice_metric", save_best_only=True, verbose=1),
        ]
    )
