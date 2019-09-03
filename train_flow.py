import pickle
import numpy as np
from i3d_inception import Inception_Inflated3d
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping
from keras.utils import multi_gpu_model


flow_train_path = ["../crossval5_8/aug_f_train"+str(i)+".p" for i in range(5)]
rgb_train_path = ["../crossval5_8/r_train"+str(i)+".p" for i in range(5)]
label_train_path = ["../crossval5_8/aug_l_train"+str(i)+".p" for i in range(5)]

def generate_arrays_from_file(data, labels):
	while True:
		for i in range(len(labels)):
			#x, y = data[i], labels[i]
			x, y = np.load(data[i]), labels[i]
			x = x.reshape((1, x.shape[0], x.shape[1], x.shape[2], x.shape[3]))
			yield x, y


earlystop = EarlyStopping(monitor='acc', min_delta=0, patience=5, verbose=0, mode='auto')

for i in range(1):

#        rgb_model = Inception_Inflated3d(include_top=False, weights='rgb_imagenet_and_kinetics', input_shape=(None, 224, 224, 3), endpoint_logit=False, classes=8)
#        sgd = SGD(lr=1e-4, decay=1e-7, momentum=0.9, nesterov=True)
#        rgb_model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
#        rgb_model.summary()

#        rgb_train_data = pickle.load(open(rgb_train_path[i], "rb"))
#        label_train_data = pickle.load(open(label_train_path[i], "rb"))
#        steps = len(label_train_data)
#        rgb_model.fit_generator(generate_arrays_from_file(rgb_train_data, label_train_data), steps_per_epoch=steps, epochs=1)


#        rgb_model.save("data/0_8/rgb"+str(i)+".h5")

#        print("RGB Model", i, "saved \n")

        flow_model = Inception_Inflated3d(include_top=False, weights='flow_imagenet_and_kinetics', input_shape=(None, 224, 224, 2), endpoint_logit=False, classes=8)
        sgd = SGD(lr=1e-4, decay=1e-7, momentum=0.9, nesterov=True)
        flow_model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
        flow_model.summary()

        flow_train_data = pickle.load(open(flow_train_path[i], "rb"))
        label_train_data = pickle.load(open(label_train_path[i], "rb"))
        steps = len(label_train_data)
        flow_model.fit_generator(generate_arrays_from_file(flow_train_data, label_train_data), steps_per_epoch=steps, epochs=1)


        flow_model.save("data/0_8/aug_flow"+str(i)+".h5")

        print("Flow Model", i, "saved \n")
