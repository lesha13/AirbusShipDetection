import tensorflow as tf
import tensorflow.keras.backend as krs


# dice coefficient is 2 times the area of overlap divided by the total number of pixels in both the images

# metrics to be evaluated by the model during training and testing
def dice_metric(y, y_hat, smooth=1):
    y_f = krs.flatten(y)
    y_hat_f = krs.flatten(y_hat)
    intersection = krs.sum(y_f * y_hat_f)
    return (2. * intersection + smooth) / (krs.sum(y_f) + krs.sum(y_hat_f) + smooth)


# loss function is the negative dice metric
def dice_loss(y, y_hat):
    return -dice_metric(y, y_hat)
