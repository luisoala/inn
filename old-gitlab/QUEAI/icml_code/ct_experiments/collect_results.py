""" Collects the prediction results of various pretrained models. """
import importlib
import os
import sys

from tqdm import tqdm

import numpy as np
from UNetDropoutModel import UNetDropout
from UNetIntervalModel import UNetInterval
from UNetProboutModel import UNetProbout


# util function, data loading
def get_data(path):
    generator_path = os.path.expanduser(path)
    generator_spec = importlib.util.spec_from_file_location("", generator_path)
    generator_module = importlib.util.module_from_spec(generator_spec)
    generator_spec.loader.exec_module(generator_module)
    test_generator = generator_module.load_test_data(1)
    return test_generator


# main function
def run():
    DATA_PATH = sys.argv[1]  # string, path to data laoder
    MODE = sys.argv[2]  # string, one of: interval, dropout, probout

    test_generator_in, test_generator_target = get_data(DATA_PATH)
    N = test_generator_in.n
    test_generator = zip(test_generator_in, test_generator_target)

    if MODE == "interval":
        unet = UNetInterval(
            "unet_single_weights.h5", "unet_interval_weights.h5"
        )
        os.makedirs(
            os.path.join("collected_results", "interval"), exist_ok=True
        )
        for counter in tqdm(range(N)):
            inp, tar = next(test_generator)
            rec = unet.single_unet.predict(inp)
            lo, hi = unet.interval_unet.predict([inp, inp])
            np.savez_compressed(
                os.path.join(
                    "collected_results",
                    "interval",
                    "mayo_test_{:03d}.npz".format(counter),
                ),
                input=inp,
                target=tar,
                reconstruction=rec,
                lo=lo,
                hi=hi,
            )
    elif MODE == "dropout":
        unet = UNetDropout("unet_only_weights.h5")
        os.makedirs(
            os.path.join("collected_results", "dropout"), exist_ok=True
        )
        for counter in tqdm(range(N)):
            inp, tar = next(test_generator)
            rec = unet.model.predict(inp)
            mean, var = unet.dropout_predict(inp, num_samples=128)
            np.savez_compressed(
                os.path.join(
                    "collected_results",
                    "dropout",
                    "mayo_test_{:03d}.npz".format(counter),
                ),
                input=inp,
                target=tar,
                reconstruction=rec,
                mean=mean,
                variance=var,
            )
    elif MODE == "probout":
        unet = UNetProbout("unet_probout_weights.h5")
        os.makedirs(
            os.path.join("collected_results", "probout"), exist_ok=True
        )
        for counter in tqdm(range(N)):
            inp, tar = next(test_generator)
            out = unet.model.predict(inp)
            mean, log_var = out[..., :1], out[..., 1:]
            var = np.exp(log_var)
            rec = mean
            np.savez_compressed(
                os.path.join(
                    "collected_results",
                    "probout",
                    "mayo_test_{:03d}.npz".format(counter),
                ),
                input=inp,
                target=tar,
                reconstruction=rec,
                mean=mean,
                variance=var,
            )
    else:
        print(
            "Unknown mode. Expected 'interval', 'dropout', "
            "or 'probout', but got {}.".format(MODE)
        )


if __name__ == "__main__":
    run()
