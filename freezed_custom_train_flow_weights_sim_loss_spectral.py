import pickle
import numpy as np
from i3d_inception import Inception_Inflated3d
from keras.optimizers import SGD, Adam
from keras.callbacks import EarlyStopping
import tensorflow as tf
from keras import backend as K
import os
from keras.callbacks import Callback
from keras.metrics import binary_accuracy
import numpy.linalg as linalg

flow_model_kin = Inception_Inflated3d(include_top=False, weights='flow_imagenet_and_kinetics', input_shape=(None, 224, 224, 2), dropout_prob=0.5,endpoint_logit=False, classes=8)
flow_model_kin.load_weights("/DATA/keras-kinetics-i3d/data/0_8/kin_flow_0.h5")

flow_train_path = "../crossval5_8/less_f_train7.p"
rgb_train_path = "../crossval5_8/less_r_train7.p"
label_train_path = "../crossval5_8/less_l_train7.p"



class Flda(Callback):
    def __init__(self, w):
        super(Flda, self).__init__()
        self.w = K.variable(w, dtype=np.float32)

    def on_batch_begin(self, batch, logs={}):
        kin_wts=flow_model_kin.layers[196].get_weights()[0]
        #kin_wts=kin_wts.reshape(8192)
        #kin_wts=kin_wts[0:8100]
        #kin_wts=kin_wts.reshape(90,90)
        #kin_wts=kin_wts/np.linalg.norm(kin_wts)
        kin_wts=kin_wts.reshape(1024,8)
        kin_wts=np.matmul(kin_wts.T,kin_wts)

        aut_wts=self.model.layers[196].get_weights()[0]
        #aut_wts=aut_wts.reshape(8192)
        #aut_wts=aut_wts[0:8100]
        #aut_wts=aut_wts.reshape(90,90)
        #aut_wts=aut_wts/np.linalg.norm(aut_wts) 
        aut_wts=aut_wts.reshape(1024,8)
        aut_wts=np.matmul(aut_wts.T,aut_wts)
        
        egnval_kin ,egnvecs_kin=np.linalg.eigh(kin_wts)
        egnval_aut ,egnvecs_aut=np.linalg.eigh(aut_wts)
        
        eigen_pairs_kin = [[np.abs(egnval_kin[i]),egnvecs_kin[:,i]] for i in range(len(egnval_kin))]
        eigen_pairs_kin = sorted(eigen_pairs_kin,key=lambda k: k[0],reverse=True)
        
        Wk=np.vstack((eigen_pairs_kin[0][1].real,eigen_pairs_kin[1][1].real)).T
        
        #print(Wk)
        eigen_pairs_aut = [[np.abs(egnval_aut[i]),egnvecs_aut[:,i]] for i in range(len(egnval_aut))]
        eigen_pairs_aut = sorted(eigen_pairs_aut,key=lambda k: k[0],reverse=True)
        
        Wa=np.vstack((eigen_pairs_aut[0][1].real,eigen_pairs_aut[1][1].real)).T
        #print(Wa)
        loss=np.linalg.norm(np.matmul(Wk.T,Wa)-np.identity(2))   
        K.set_value(self.w,loss)
        #print(loss)
#a=[Move_the_table,'Touchear','LockHands',Touchhead,Touchnose,Rolly_Polly,Tapping, 'Arms_Up']

def flda_loss(y_true,y_pred):
        return (K.categorical_crossentropy(y_true, y_pred)+flda_obj.w)
    

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
        flda_obj=Flda(0.0)
        flow_model = Inception_Inflated3d(include_top=False, weights='flow_imagenet_and_kinetics', input_shape=(None, 224, 224, 2), dropout_prob=0.5,endpoint_logit=False, classes=8)
        flow_model.load_weights("/DATA/keras-kinetics-i3d/data/0_8/n_flow_wts_7.h5")


        #flow_model.load_weights("/DATA/keras-kinetics-i3d/data/0_8/53_mm_weights_sim_20e_0.h5")
        adam = Adam(lr=1e-4, decay=0)
        for j,layer in enumerate(flow_model.layers):
                if(j<53):
                        layer.trainable = False
                else:
                        layer.trainable = True
        inputl = flow_model.inputs
        print(inputl)
        flow_model.compile(loss=flda_loss, optimizer=adam, metrics=['accuracy'])
        flow_model.summary()
        
        flow_train_data = pickle.load(open(flow_train_path, "rb"))
        label_train_data = pickle.load(open(label_train_path, "rb"))
        steps = len(label_train_data)
        #bools = pickle.load( open( "/DATA/keras-kinetics-i3d/bool_mm_3c_20e.p", "rb" ) )
        hist=flow_model.fit_generator(generate_arrays_from_file(flow_train_data, label_train_data), steps_per_epoch=steps, epochs=10,callbacks=[flda_obj])
        
        flow_model.save("data/0_8/ablation_without_dmm_10e_"+str(i)+".h5")

        print("Flow Model", i, "saved \n")


