from UNetIntervalModel import UNetInterval

import os, sys
import importlib

import matplotlib.pyplot as plt


# SETUP PARAMETERS
DATA_PATH = sys.argv[1]     # should be a string
EPOCHS = int(sys.argv[2])   # should be integer
BATCH_SIZE = int(sys.argv[3])   # should be integer
TRAIN_MODE = sys.argv[4]    # should be a string (either single or interval or
                            # single-continue or interval-continue)

# UTIL FUNCTIONS
def get_data(path):
    generator_path = os.path.expanduser(path)
    generator_spec = importlib.util.spec_from_file_location('', generator_path)
    generator_module = importlib.util.module_from_spec(generator_spec)
    generator_spec.loader.exec_module(generator_module)
    train_generator = generator_module.load_train_data(BATCH_SIZE)
    val_generator = generator_module.load_val_data(BATCH_SIZE)
    return train_generator, val_generator

def double_generator(gen):
    while True:
        tmp = next(gen)
        yield [tmp, tmp]

# RUN TRAINING
train_generator, val_generator = get_data(DATA_PATH)
train_generator_in, train_generator_target = train_generator
val_generator_in, val_generator_target = val_generator
if TRAIN_MODE == 'single':
    unet_interval = UNetInterval()
    unet_interval.train(
        EPOCHS,
        BATCH_SIZE,
        zip(train_generator_in, train_generator_target),
        zip(val_generator_in, val_generator_target),
        verbose=0,
    )
    unet_interval.single_unet.save_weights('unet_single_weights.h5')
elif TRAIN_MODE == 'single-continue':
    unet_interval = UNetInterval('unet_single_weights_200.h5')
    unet_interval.train(
        EPOCHS,
        BATCH_SIZE,
        zip(train_generator_in, train_generator_target),
        zip(val_generator_in, val_generator_target),
        verbose=0,
    )
    unet_interval.single_unet.save_weights('unet_single_continued_weights.h5')
elif TRAIN_MODE == 'interval':
    unet_interval = UNetInterval('unet_single_weights_200.h5')
    unet_interval.train_interval(
        EPOCHS,
        BATCH_SIZE,
        zip(double_generator(train_generator_in),
            double_generator(train_generator_target)),
        zip(double_generator(val_generator_in),
            double_generator(val_generator_target)),
        verbose=0,
    )
    unet_interval.interval_unet.save_weights('unet_interval_weights.h5')
elif TRAIN_MODE == 'interval-continue':
    pass
    # TODO
