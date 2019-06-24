import matplotlib.pyplot as plt
plt.ioff() 
import pickle
import numpy as np
import pandas as pd
from keras.models import Sequential, Model
from keras.layers import Dense,Dropout,Input
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

def SiDNN(FrontLayers):
	data=pickle.load(open('run_100_60_10000bigoutput.txt','rb'))
	data = data[0]
	cols = []
	NLayers = len(data[0])
	for i in range(NLayers):
		cols.append('layer'+str(i))
	df = pd.DataFrame(data=data,columns=cols)
	#FrontLayers = 6 
	DropLayers = []
	for i in range(FrontLayers,NLayers):
		DropLayers.append('layer'+str(i))
		#Array to drop everything that we're not including in the front
	leakLayers =[]
	LeakageLayerStart = 40 #Only count first 40 layers (0-39)
	for i in range(LeakageLayerStart,NLayers):
		leakLayers.append('layer'+str(i))
	leakage=df[leakLayers].sum(axis=1) #Calculate Leakage and make dataframe for comparison as truth
	df = df.drop(DropLayers,axis=1) #Remove Leakage Layers from main data frame
	
	# Preprocessing to make our model happy, would be needed to be used again if called outside of this program. I.E. to implement the neural network as it only works with the scaled information
	scalerFront,scalerLeakage= RobustScaler(),RobustScaler()
	#scalerFront.fit_transform(df)
	truthVals=np.asarray(leakage).reshape(len(leakage),1)
	#scalerLeakage.fit_transform(truthVals)
	
	
	
############################
########## MODEL ###########
############################
	def DNNmodel(Input_shape=10, n_hidden=1, n_nodesHidden=10, dropout=0.2, optimizer='adam'):
	        model = Sequential()
		model.add(Dense(Input_shape,input_dim=Input_shape,activation='relu'))
	#	inputs=Input(shape=Input_shape)
        	i=0
	        if n_hidden>0:
	#                hidden=Dense(n_nodesHidden, activation='relu')(inputs)
	#                hidden=Dropout(dropout)(hidden)
			model.add(Dense(n_nodesHidden,activation='relu'))
			model.add(Dropout(dropout))
	                i+=1
	        while i<n_hidden:
	#                hidden=Dense(n_nodesHidden, activation='relu')(hidden)
	#                hidden=Dropout(dropout)(hidden)
			model.add(Dense(n_nodesHidden,activation='relu'))
			model.add(Dropout(dropout))
	                i+=1
	#        outputs = Dense(1,activation='linear')(hidden)
		model.add(Dense(1,activation='linear'))
	#        model = Model(inputs,outputs)
	        model.compile(optimizer=optimizer, loss='logcosh', metrics=['mean_absolute_error'])
	        model.summary()
	        return model
	model=DNNmodel(Input_shape=FrontLayers,n_hidden=3,n_nodesHidden=20)
	
	print "Model Compiled"
	
	ix=range(leakage.shape[0])
	truth_train,truth_val,layers_train,layers_val,ix_train,ix_val = train_test_split(truthVals,df,ix,test_size=0.1)
	#truth_train,truth_test,layers_train,layers_test,ix_train,ix_test = train_test_split(truthVals,df,ix,test_size=0.1)
	#truth_train,truth_val,layers_train,layers_val,ix_train,ix_val   = train_test_split(truth_train, layers_train, ix_train,test_size=0.1)

	print "Training: "
	try:
		model.fit(
			layers_train,truth_train,
			callbacks = [
			    EarlyStopping(verbose=True,patience=200,monitor='val_loss'),
			    ModelCheckpoint(str(FrontLayers)+'FrontLayersmodelBest.h5',monitor='val_loss',verbose=True, save_best_only=True),
			    ],
			epochs=4000,
			batch_size=10,
			validation_data=(layers_val,truth_val)
	)
	except KeyboardInterrupt:
		print "Training ended early via KeyboardInterrupt"
	model.load_weights(str(FrontLayers)+'FrontLayersmodelBest.h5')
	
	###################################################
	### Some more plotting
	#plt.clf()
	plt.scatter(np.asarray(leakage).reshape(len(leakage),1),model.predict(df), c='Black',s=0.1)
	plt.plot(np.linspace(0,250,10),np.linspace(0,250,10),c='red',marker='',linestyle='-')
	plt.xlabel('True Leakage Values, MeV (Sum Layers %i-60)'%LeakageLayerStart)
	plt.ylabel('NN Prediction, MeV, Using Layers 0-%i'%FrontLayers)
	plt.tight_layout()
	plt.xlim(0,100)
	plt.ylim(0,100)
	plt.savefig(str(FrontLayers)+'FrontLayersPredictionVsTruth.png')
	plt.xlim(0,30)
	plt.ylim(0,30)
	plt.savefig(str(FrontLayers)+'FrontLayersPredictionVsTruthSmall.png')
#	plt.show()
	plt.clf()
	print "PredictionVsTruth.png Written"
	
	plt.scatter(df.sum(axis=1),model.predict(df),c='Black',s=0.1,label='ModelPrediction')
	plt.scatter(df.sum(axis=1),np.asarray(leakage).reshape(len(leakage),1),c='Red',s=0.1,label='Leakage',alpha=0.4)
	plt.xlabel("Energy in Front Layers 0-%i"%FrontLayers)
	plt.ylabel('NN Prediction, Actual Leakage')
	plt.legend()
	plt.savefig(str(FrontLayers)+'FrontLayersLeakageFrontEnergy.png')
	plt.clf()
	print "Sum of weights of first layer mapped to input variable: "
	we = model.layers[1].get_weights()
	for i in range(len(we[0])):
	        print "Layer ",i, ": ", sum(we[0][i])

if __name__ == "__main__":
	for i in range(1,41):
		SiDNN(i)
