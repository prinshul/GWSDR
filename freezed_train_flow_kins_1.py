import pickle
import numpy as np
from i3d_inception import Inception_Inflated3d
from keras.optimizers import SGD, Adam
from keras.callbacks import EarlyStopping
from sklearn.utils import class_weight

flow_train_path = "../crossval5_8/5050_kin_f_train0.p" 
rgb_train_path = "../crossval5_8/r_train0.p" 
label_train_path = "../crossval5_8/5050_kin_l_train0.p" 

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


        #rgb_model.save("data/0_8/rgb"+str(i)+".h5")

#        print("RGB Model", i, "saved \n")

        flow_model = Inception_Inflated3d(include_top=False, weights='flow_imagenet_and_kinetics', input_shape=(None, 224, 224, 2), dropout_prob=0.5,endpoint_logit=False, classes=8)
        #flow_model.load_weights("/DATA/keras-kinetics-i3d/data/0_8/n_flow_wts_7.h5")
        #flow_model.load_weights("data/0_8/53_mm_all_3c_20e_0.h5")
        adam = Adam(lr=1e-4, decay=0)
        for j,layer in enumerate(flow_model.layers):
                if(j<196):
                        layer.trainable = False
                else:
                        layer.trainable = True
        flow_model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
        flow_model.summary()

        flow_train_data = pickle.load(open(flow_train_path, "rb"))
        label_train_data = pickle.load(open(label_train_path, "rb"))
        steps = len(label_train_data)
        y_lab=np.concatenate(label_train_data, axis=0 )
        y_lab=np.argmax(y_lab,axis=1)
        flow_model.fit_generator(generate_arrays_from_file(flow_train_data, label_train_data), steps_per_epoch=steps, epochs=10)

        #flow_model.fit_generator(generate_arrays_from_file(flow_train_data, label_train_data), steps_per_epoch=steps,class_weight=class_weights, epochs=10)
        flow_model.save("data/0_8/kin_flow_5050_80_"+str(i)+".h5")

        print("Flow Model", i, "saved \n")
