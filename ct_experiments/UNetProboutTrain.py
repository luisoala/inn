""" Trains (refines) a Probout U-Net. """
import importlib
import os
import sys

from UNetProboutModel import UNetProbout

# SETUP PARAMETERS
DATA_PATH = sys.argv[1]  # should be a string
EPOCHS = int(sys.argv[2])  # should be an integer
BATCH_SIZE = int(sys.argv[3])  # should be an integer


# UTIL FUNCTIONS
def get_data(path):
    generator_path = os.path.expanduser(path)
    generator_spec = importlib.util.spec_from_file_location("", generator_path)
    generator_module = importlib.util.module_from_spec(generator_spec)
    generator_spec.loader.exec_module(generator_module)
    train_generator = generator_module.load_train_data(BATCH_SIZE)
    val_generator = generator_module.load_val_data(BATCH_SIZE)
    return train_generator, val_generator


# RUN TRAINING
train_generator, val_generator = get_data(DATA_PATH)
train_generator_in, train_generator_target = train_generator
val_generator_in, val_generator_target = val_generator

unet_probout = UNetProbout("unet_only_weights.h5", lr=1e-7, split=None)
unet_probout.train(
    EPOCHS,
    BATCH_SIZE,
    zip(train_generator_in, train_generator_target),
    zip(val_generator_in, val_generator_target),
    verbose=0,
)
unet_probout.model.save_weights("unet_probout_weights.h5")
