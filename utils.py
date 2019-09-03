import os
import sys
import pickle
import numpy as np
from random import shuffle
import random

classes = os.listdir('../autism_rgb/')
print(classes)
pickle.dump(classes, open("classes.p", "wb"))
#sys.exit()
classfiles = [os.listdir('../autism_flow/'+classes[i]) for i in range(len(classes))]
print([len(i) for i in classfiles])
classes_dict = {i:classes[i] for i in range(len(classes))}
print(classes_dict)


def chunk_5(data):
        print(len(data))
        s = (len(data)/8)*4
        #s=random.sample([92,93],1)[0]
        #s=random.sample([90,91,92,93,94,95],1)[0]
        l = [data[int(i*s):int(i*s+s)] for i in range(8)]
        return l

for i in range(len(classes)):
        shuffle(classfiles[i])
        classfiles[i] = chunk_5(classfiles[i])
        #print(len(classfiles[i]))


#print([len(classfiles[i]) for i in range(len(classfiles))])
# sys.exit()

for t in range(8):
        flow_train = []
        rgb_train = []
        labels_train = []
        flow_test = []
        rgb_test = []
        labels_test = []
        for i in range(len(classes)):
                train_classfiles = classfiles[i][t]
                #test_classfiles = classfiles[i][t]
                print('train',len(train_classfiles))
                test_classfiles = []
                for tt in range(8):
                        if tt != t:
                                test_classfiles = test_classfiles + classfiles[i][tt]
                
                for file in train_classfiles:
                        
                        x = '../autism_flow/'+classes[i]+'/'+file
                        flow_train.append(x)
                        
                        x = '../autism_rgb/'+classes[i]+'/'+file
                        rgb_train.append(x)
                        y = np.zeros((1, 8))
                        y[0, i] = 1
                        labels_train.append(y)
                for file in test_classfiles:
                        
                        x = '../autism_flow/'+classes[i]+'/'+file
                        flow_test.append(x)
                        
                        x = '../autism_rgb/'+classes[i]+'/'+file
                        rgb_test.append(x)
                        y = np.zeros((1, 8))
                        y[0, i] = 1
                        labels_test.append(y)
        print("data split done -", t)
	
        flow_shuf_train = []
        rgb_shuf_train = []
        labels_shuf_train = []
        flow_shuf_test = []
        rgb_shuf_test = []
        labels_shuf_test = []
        index_shuf = [i for i in range(len(labels_train))]
        shuffle(index_shuf)
        flow_shuf_train = [flow_train[i] for i in index_shuf]
        rgb_shuf_train = [rgb_train[i] for i in index_shuf]
        labels_shuf_train = [labels_train[i] for i in index_shuf]
        index_shuf = [i for i in range(len(labels_test))]
        shuffle(index_shuf)
        flow_shuf_test = [flow_test[i] for i in index_shuf]
        rgb_shuf_test = [rgb_test[i] for i in index_shuf]
        labels_shuf_test = [labels_test[i] for i in index_shuf]

        print(len(flow_shuf_train))
        print(len(flow_shuf_test))
        print(len(rgb_shuf_train))
        print(len(rgb_shuf_test))

        pickle.dump(labels_shuf_train, open("../crossval5_8/5050_l_train"+str(t)+".p", "wb"))
        pickle.dump(flow_shuf_train, open("../crossval5_8/5050_f_train"+str(t)+".p", "wb"))
        pickle.dump(rgb_shuf_train, open("../crossval5_8/5050_r_train"+str(t)+".p", "wb"))
        pickle.dump(labels_shuf_test , open("../crossval5_8/5050_l_test"+str(t)+".p", "wb"))
        pickle.dump(flow_shuf_test, open("../crossval5_8/5050_f_test"+str(t)+".p", "wb"))
        pickle.dump(rgb_shuf_test, open("../crossval5_8/5050_r_test"+str(t)+".p", "wb"))
        print("data created :", t)
