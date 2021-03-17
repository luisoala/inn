import numpy as np
import matplotlib.pyplot as plt
import sys

def cosine_distance(x,y):
	return np.sum(x*y)/(np.linalg.norm(x.ravel())*np.linalg.norm(y.ravel()))

def test_mse(x,y):
	return np.mean((x-y)**2)


runs = sys.argv[1].split(',') #comma separated enumeration of runs, e.g. run00,run01,run02,run03 ...

for method in ['interval', 'probout', 'dropout']:
	cos_collect = []
	pwcc_collect = []
	mse_collect = []
	for run in runs:
		cos_similarity = 0
		pwcc = 0
		mse = 0
		for i in range(330):
			test_sample = np.load('data/'+run+'/'+method+'/'+'mayo_test_{:03d}.npz'.format(i))
			if method == 'interval':			
				uncertainty = (test_sample['hi']-test_sample['lo'])[0,:,:,0]
			else:
				uncertainty = (test_sample['variance'][0,:,:,0])**0.5
			abs_error = np.abs(test_sample['reconstruction']-test_sample['target'])[0,:,:,0]
			
			a = cosine_distance(abs_error, uncertainty)
			b = test_mse(test_sample['target'], test_sample['reconstruction'])
			cos_similarity += a
			pwcc += a/b
			mse += b

		cos_similarity /= 330
		pwcc /= 330
		mse /= 330

		cos_collect.append(cos_similarity)
		pwcc_collect.append(pwcc)
		mse_collect.append(mse)
		print(run+' done!')

	cos_collect = np.array(cos_collect)
	pwcc_collect = np.array(pwcc_collect)
	mse_collect = np.array(mse_collect)	
	
	print(method)
	print('all cos:', cos_collect)
	print('mean cos:', np.mean(cos_collect))
	print('std cos:', np.std(cos_collect))

	print('all pwcc:', pwcc_collect)
	print('mean pwcc:', np.mean(pwcc_collect))
	print('std pwcc:', np.std(pwcc_collect))
	
	print('all mse:', mse_collect)
	print('mean mse:', np.mean(mse_collect))
	print('std mse:', np.std(mse_collect))
