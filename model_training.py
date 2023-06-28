import os
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from model_build import UNet
from train_test_dataset import get_train_dataset, get_validation_dataset
from dice_coefficient import dice_metric, dice_loss

# define paths to files
PATH = "./"
MODEL_PATH = os.path.join(PATH, "model.h5")

# other constants
EPOCHS = 10
LR = 1e-3


def train():
    # get train and validation dataset
    train_dataset = get_train_dataset()
    validation_dataset = get_validation_dataset()

    # initialize model
    model = UNet()

    # compile model using dice coefficient and Adam optimizer
    model.compile(
        optimizer=Adam(LR),
        loss=[dice_loss],
        metrics=[dice_metric],
    )

    # model training
    model.fit(
        # no need to pass y if dataset is a subclass of Sequence
        x=train_dataset,
        validation_data=validation_dataset,
        epochs=EPOCHS,
        callbacks=[
            # saves best model every epoch
            ModelCheckpoint(MODEL_PATH, monitor="dice_metric", save_best_only=True, verbose=1),
            # stops the training if there is no improvement
            EarlyStopping(patience=3),
            # reduces learning rate if there in no improvement
            ReduceLROnPlateau(patience=2),
        ]
    )
