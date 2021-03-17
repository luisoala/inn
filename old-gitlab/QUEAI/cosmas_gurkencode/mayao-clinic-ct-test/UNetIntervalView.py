from UNetIntervalModel import UNetInterval

import os, sys
import importlib
import numpy as np
import matplotlib.pyplot as plt


# SETUP PARAMETERS
DATA_PATH = sys.argv[1]     # should be a string
NUMBER = int(sys.argv[2])   # should be integer
MODE = sys.argv[3]          # should be a string (either single or interval)
VIEWMODE = sys.argv[4]      # should be a string (either show or save)

# UTIL FUNCTIONS
def get_data(path):
    generator_path = os.path.expanduser(path)
    generator_spec = importlib.util.spec_from_file_location('', generator_path)
    generator_module = importlib.util.module_from_spec(generator_spec)
    generator_spec.loader.exec_module(generator_module)
    test_generator = generator_module.load_train_data(1)
    return test_generator

# RUN TRAINING
test_generator_in, test_generator_target = get_data(DATA_PATH)
test_generator = zip(test_generator_in, test_generator_target)
if MODE == 'single':
    unet_interval = UNetInterval('unet_single_weights_200.h5')
    if VIEWMODE == 'show':
        for k in range(NUMBER):
            inp, tar = next(test_generator)
            rec = unet_interval.single_unet.predict(inp)
            plt.figure()
            plt.subplot(1, 3, 1)
            plt.imshow(tar.squeeze(), vmin=0.0, vmax=1.0)
            plt.title('original')  
            plt.subplot(1, 3, 2)
            plt.imshow(inp.squeeze(), vmin=0.0, vmax=1.0)
            plt.title('input')  
            plt.subplot(1, 3, 3)
            plt.imshow(rec.squeeze(), vmin=0.0, vmax=1.0)
            plt.title('reconstruction')
        plt.show()
    elif VIEWMODE == 'save':
        os.makedirs('saved_images', exist_ok=True)
        for k in range(NUMBER):
            inp, tar = next(test_generator)
            rec = unet_interval.single_unet.predict(inp)
            plt.imsave(
                os.path.join(
                    'saved_images',  
                    'unet_single_im{}_input.png'.format(k),
                ),
                inp.squeeze(),
                cmap='Greys_r',
                vmin=0.0,
                vmax=1.0,
                format='png',
            )
            plt.imsave(
                os.path.join(
                    'saved_images',  
                    'unet_single_im{}_target.png'.format(k),
                ),
                tar.squeeze(),
                cmap='Greys_r',
                vmin=0.0,
                vmax=1.0,
                format='png',
            )
            plt.imsave(
                os.path.join(
                    'saved_images',  
                    'unet_single_im{}_reconstruction.png'.format(k),
                ),
                rec.squeeze(),
                cmap='Greys_r',
                vmin=0.0,
                vmax=1.0,
                format='png',
            )
            plt.imsave(
                os.path.join(
                    'saved_images',  
                    'unet_single_im{}_error.png'.format(k),
                ),
                np.abs(tar-rec).squeeze(),
                cmap='Reds',
                vmin=0.0,
                vmax=1.0,
                format='png',
            )
elif MODE == 'interval':
    unet_interval = UNetInterval('unet_single_weights_200.h5', 'unet_interval_weights.h5')
    if VIEWMODE == 'show':
        for k in range(NUMBER):
            inp, tar = next(test_generator)
            rec = unet_interval.single_unet.predict(inp)
            lo, hi = unet_interval.interval_unet.predict([inp, inp])
            plt.figure()
            plt.subplot(2, 3, 1)
            plt.imshow(tar.squeeze(), vmin=0.0, vmax=1.0)
            plt.title('original')  
            plt.subplot(2, 3, 2)
            plt.imshow(inp.squeeze(), vmin=0.0, vmax=1.0)
            plt.title('input')  
            plt.subplot(2, 3, 3)
            plt.imshow((hi-lo).squeeze())
            plt.title('max-min')  
            plt.subplot(2, 3, 4)
            plt.imshow(rec.squeeze(), vmin=0.0, vmax=1.0)
            plt.title('reconstruction')  
            plt.subplot(2, 3, 5)
            plt.imshow(lo.squeeze(), vmin=0.0, vmax=1.0)
            plt.title('min')  
            plt.subplot(2, 3, 6)
            plt.imshow(hi.squeeze(), vmin=0.0, vmax=1.0)
            plt.title('max')    
        plt.show()
    elif VIEWMODE == 'save':
        os.makedirs('saved_images', exist_ok=True)
        for k in range(NUMBER):
            inp, tar = next(test_generator)
            rec = unet_interval.single_unet.predict(inp)
            lo, hi = unet_interval.interval_unet.predict([inp, inp])    
            plt.imsave(
                os.path.join(
                    'saved_images',  
                    'unet_interval_im{}_input.png'.format(k),
                ),
                inp.squeeze(),
                cmap='Greys_r',
                vmin=0.0,
                vmax=1.0,
                format='png',
            )
            plt.imsave(
                os.path.join(
                    'saved_images',  
                    'unet_interval_im{}_target.png'.format(k),
                ),
                tar.squeeze(),
                cmap='Greys_r',
                vmin=0.0,
                vmax=1.0,
                format='png',
            )
            plt.imsave(
                os.path.join(
                    'saved_images',  
                    'unet_interval_im{}_reconstruction.png'.format(k),
                ),
                rec.squeeze(),
                cmap='Greys_r',
                vmin=0.0,
                vmax=1.0,
                format='png',
            )
            plt.imsave(
                os.path.join(
                    'saved_images',  
                    'unet_interval_im{}_error.png'.format(k),
                ),
                np.abs(tar-rec).squeeze(),
                cmap='Reds',
                vmin=0.0,
                vmax=1.0,
                format='png',
            )
            plt.imsave(
                os.path.join(
                    'saved_images',  
                    'unet_interval_im{}_lower.png'.format(k),
                ),
                lo.squeeze(),
                cmap='Greys_r',
                vmin=0.0,
                vmax=1.0,
                format='png',
            )
            plt.imsave(
                os.path.join(
                    'saved_images',  
                    'unet_interval_im{}_upper.png'.format(k),
                ),
                hi.squeeze(),
                cmap='Greys_r',
                vmin=0.0,
                vmax=1.0,
                format='png',
            )
            plt.imsave(
                os.path.join(
                    'saved_images',  
                    'unet_interval_im{}_interval.png'.format(k),
                ),
                (hi-lo).squeeze(),
                cmap='Reds',
                vmin=0.0,
                vmax=1.0,
                format='png',
            )
