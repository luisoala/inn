import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

import os 
dir_path = os.path.dirname(os.path.realpath(__file__))

mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']

figlegend = plt.figure(figsize=(32,6))
handles = []
handles.append(mpatches.Patch(color='green', label=r'$\mathrm{Input}$'))
handles.append(mpatches.Patch(color=[0,0,1,0.5], label=r'$\mathrm{Uncertainty}$'))
handles.append(mpatches.Patch(color='black', label=r'$\mathrm{Target}$'))
handles.append(mpatches.Patch(color=[0.9290, 0.6940, 0.1250], label=r'$\mathrm{Model\ Output}$'))
handles.append(mpatches.Patch(color='maroon', label=r'$\mathrm{Absolute\ Error}$'))

figlegend.legend(handles = handles, fontsize=55, mode='expand', ncol=5)
plt.savefig('legend.png')
plt.close()


image_files = [el for el in os.listdir(dir_path) if el[-4:]=='.png']

for fname in image_files:
	img = Image.open(fname)
	im_ar = np.array(img)
	#print(im_ar.shape)
	x_white = np.all(im_ar==255, axis=(1,2))
	while_rows = np.logical_not(x_white).nonzero()[0]
	a,b = while_rows[0], while_rows[-1]+1

	y_white = np.all(im_ar==255, axis=(0,2))
	while_columns = np.logical_not(y_white).nonzero()[0]
	c,d = while_columns[0], while_columns[-1]+1

	new_im = img.crop(box=(c,a,d,b))

	new_im.save(fname)


image_files = [el for el in os.listdir(dir_path) if el[-4:]=='.png' and ('Noisy' in el) and ('legend' not in el)]

for fname in image_files:
	fname = fname.replace('Noisy','')
	img = Image.open(fname)
	im_ar = np.array(img)
	#print(im_ar.shape)
	h1, b1 = im_ar.shape[:2]

	f_name_parts = fname.split('_')
	img = Image.open(f_name_parts[0]+'Noisy_'+f_name_parts[1])
	im_ar = np.array(img)
	#print(im_ar.shape)
	h2, b2 = im_ar.shape[:2]
	im_ar = np.pad(im_ar, ((h1-h2,0),(0,b1-b2),(0,0)), mode='constant', constant_values=255)
	new_img = Image.fromarray(im_ar)
	new_img.save(f_name_parts[0]+'Noisy_'+f_name_parts[1])
