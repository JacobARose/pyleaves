
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD



img_shape = (299,299)
preprocess = preprocess_input



def build_model(num_classes, learning_rate):

	base_model = InceptionV3(weights='imagenet', include_top=True)
	
	logits = Dense(num_classes, name='logits')(base_model.layers[-2].output)
	predictions = Activation('softmax',name='predictions')(logits)
	model = Model(inputs=base_model.input, outputs=predictions)
	
	opt = SGD(lr=learning_rate)
	model.compile(optimizer=opt,
					loss='categorical_crossentropy',
					metrics=['accuracy'])
					
	return model

def train_model(model,
				train_data,
				validation_data=None, 
				steps_per_epoch=None, 
				validation_steps=None, 
				max_epochs=None, 
				callbacks=None,
				workers=-1,
				initial_epoch=0,
				verbose=True):
	
	history = model.fit(
						train_data,
						steps_per_epoch=steps_per_epoch,
						epochs=max_epochs,
						validation_data=validation_data,
						validation_steps=validation_steps,
						callbacks=callbacks,
						workers=-1,
						initial_epoch=initial_epoch,
						verbose=verbose)
	return history