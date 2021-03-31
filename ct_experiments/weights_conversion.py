""" Extracts the underlying single U-Net weights of an Interval U-Net. """
import shutil

import h5py
import numpy as np

IN_FILE = "unet_single_weights.h5"
OUT_FILE = "unet_only_weights.h5"

shutil.copyfile(IN_FILE, OUT_FILE)
f = h5py.File(OUT_FILE, "r+")

forbidden_words = [
    "min_bias",
    "max_bias",
    "min_kernel",
    "max_kernel",
]


def delete_forbidden(a):
    print(list(f[a].attrs))
    if "weight_names" in f[a].attrs:
        for name in f[a].attrs["weight_names"]:
            if any(forbidden in str(name) for forbidden in forbidden_words):
                print("removing {}".format(name))
                f[a].attrs["weight_names"] = np.delete(
                    f[a].attrs["weight_names"],
                    np.where(f[a].attrs["weight_names"] == name),
                )
                del f[a][name]


print("====== PRE =====")
f.visit(print)  # noqa: E999
f.visit(
    lambda a: print(
        f[a].attrs["weight_names"] if "weight_names" in f[a].attrs else None
    )
)
f.visit(delete_forbidden)
print("====== POST =====")
f.visit(print)
f.visit(
    lambda a: print(
        f[a].attrs["weight_names"] if "weight_names" in f[a].attrs else None
    )
)
