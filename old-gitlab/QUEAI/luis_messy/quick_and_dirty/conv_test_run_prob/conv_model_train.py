from conv_model import *
from utils import *
import sys



safe_mkdir('matrices-train')
safe_mkdir('matrices-test')

DATA_PATH = sys.argv[1] #should be a string

MODEL_TYPE = sys.argv[2] #'prob', 'drop', 'det', 'error_pred'

EPOCHS = int(sys.argv[3])

BATCH_SIZE = int(sys.argv[4])

SAMPLING_INTERVAL = int(sys.argv[5])
	

#training
if MODEL_TYPE == 'error_pred':
	X_train, Y_train, X_val, Y_val, X_test, Y_test =  get_data(DATA_PATH) #load data for original prediction task
	convnet_pred = ConvNet_Det() #inititalize a prediction network instance
	convnet_pred.convnet.load_weights('convnet_det.h5', by_name=True) # load weights of pretrained prediction network
	#predict on train, val and test
	Preds_train = convnet_pred.convnet.predict(X_train)
	Preds_val = convnet_pred.convnet.predict(X_val)
	Preds_test = convnet_pred.convnet.predict(X_test)
	#get errors
	Error_train = np.abs(Y_train - Preds_train)
	Error_val = np.abs(Y_val - Preds_val)
	Error_test = np.abs(Y_test - Preds_test)
	
	#build error pred inputs from prediction model inputs and outputs
	disc_input_train = np.concatenate([X_train, Preds_train], axis=-1)
	disc_input_val = np.concatenate([X_val, Preds_val], axis=-1)
	disc_input_test = np.concatenate([X_test, Preds_test], axis=-1)
	
	K.clear_session() #clear memory
	
	convnet_errorpred = ConvNet_ErrorPred() #intitialize an errorprediction network instance
	
	convnet_errorpred.train(EPOCHS,BATCH_SIZE, SAMPLING_INTERVAL, disc_input_train, Error_train, disc_input_val, Error_val, disc_input_test, Error_test, True, True, 0) #train error prediction network
	

elif MODEL_TYPE == 'error_pred_propper':
	X_train, Y_train, X_val, Y_val, X_test, Y_test =  get_data(DATA_PATH) #load data for original prediction task
	convnet_pred = ConvNet_Det() #inititalize a prediction network instance
	convnet_errorpred = ConvNet_ErrorPred() #intitialize an errorprediction network instance
	
	for i in range(EPOCHS):
		convnet_pred.train(1, BATCH_SIZE, SAMPLING_INTERVAL, X_train, Y_train, X_val, Y_val, X_test, Y_test, False, False)
		#, save_params, save_samples
		#convnet_pred.convnet.load_weights('convnet_det.h5', by_name=True) # load weights of pretrained prediction network
		#predict on train, val and test
		Preds_train = convnet_pred.convnet.predict(X_train)
		Preds_val = convnet_pred.convnet.predict(X_val)
		Preds_test = convnet_pred.convnet.predict(X_test)
		#get errors
		Error_train = np.abs(Y_train - Preds_train)
		Error_val = np.abs(Y_val - Preds_val)
		Error_test = np.abs(Y_test - Preds_test)
	
		#build error pred inputs from prediction model inputs and outputs
		disc_input_train = np.concatenate([X_train, Preds_train], axis=-1)
		disc_input_val = np.concatenate([X_val, Preds_val], axis=-1)
		disc_input_test = np.concatenate([X_test, Preds_test], axis=-1)
	
		#K.clear_session() #clear memory
	
		convnet_errorpred.train(EPOCHS,BATCH_SIZE, 100, disc_input_train, Error_train, disc_input_val, Error_val, disc_input_test, Error_test, True, True, i) #train error prediction network

else:
	if MODEL_TYPE == 'det':
		convnet = ConvNet_Det()

	elif MODEL_TYPE == 'drop':
		convnet = ConvNet_Drop()
	
	elif MODEL_TYPE == 'prob':
		convnet = ConvNet_Prob()

	else:
		print('No model specified')
	
	X_train, Y_train, X_val, Y_val, X_test, Y_test =  get_data(DATA_PATH)
	convnet.train(EPOCHS, BATCH_SIZE, SAMPLING_INTERVAL, X_train, Y_train, X_val, Y_val, X_test, Y_test, True, True)
