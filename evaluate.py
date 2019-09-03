import pickle
import numpy as np
from keras.models import load_model
from i3d_inception import Inception_Inflated3d

flow_test_path = ["../crossval5_8/f_test"+str(i)+".p" for i in range(5)]
rgb_test_path = ["../crossval5_8/r_test"+str(i)+".p" for i in range(5)]
label_test_path = ["../crossval5_8/l_test"+str(i)+".p" for i in range(5)]

def generate_arrays_from_file(data_path, labels_path):
        while True:
                data = pickle.load(open(data_path, "rb"))[0]
                labels = pickle.load(open(labels_path, "rb"))[0]
                for i in range(len(labels)):
                        #x, y = data[i], labels[i]
                        x, y = np.load(data[i]), labels[i]
                        x = x.reshape((1, x.shape[0], x.shape[1], x.shape[2], x.shape[3]))
                        yield x, y

rgb_data = []
flow_data = []
labels = []
for t in range(5):
        rgb_data.append(pickle.load(open(rgb_test_path[t], "rb")))
        flow_data.append(pickle.load(open(flow_test_path[t], "rb")))
        labels.append(pickle.load(open(label_test_path[t], "rb")))

acc_list = []

for t in range(1):
        rgb_model = Inception_Inflated3d(include_top=False, weights='rgb_imagenet_and_kinetics', input_shape=(None, 224, 224, 3), classes=8)
        #rgb_model.load_weights("data/0_8/rgb"+str(t)+".h5")
        rgb_model.load_weights("data/0_8/rgb"+str(t)+".h5")
        flow_model = Inception_Inflated3d(include_top=False, weights='flow_imagenet_and_kinetics', input_shape=(None, 224, 224, 2), classes=8)
        flow_model.load_weights("data/0_8/flow"+str(t)+".h5")
        count = 0
        y_pred = []
        y_true = []
        overall_conf = []
        correct_overall_conf = []
        wrong_overall_conf = []
        label_conf = {0:[], 1:[], 2:[], 3:[], 4:[], 5:[], 6:[], 7:[]}
        correct_label_conf = {0:[], 1:[], 2:[], 3:[], 4:[], 5:[], 6:[], 7:[]}
        wrong_label_conf = {0:[], 1:[], 2:[], 3:[], 4:[], 5:[], 6:[], 7:[]}
        for i in range(len(labels[t])):
                x = np.load(rgb_data[t][i])
                x = x.reshape((1, x.shape[0], x.shape[1], x.shape[2], x.shape[3]))
                rgb_logits = rgb_model.predict(x)
                x = np.load(flow_data[t][i])
                x = x.reshape((1, x.shape[0], x.shape[1], x.shape[2], x.shape[3]))
                flow_logits = flow_model.predict(x)
                sample_logits = rgb_logits + flow_logits
                sample_logits = sample_logits[0]
                sample_predictions = np.exp(sample_logits) / np.sum(np.exp(sample_logits))
                conf = np.amax(sample_predictions)
                sorted_indices = np.argsort(sample_predictions)[::-1]
                pred_class = sorted_indices[0]
                #print(pred_class)
                true_class = np.argmax(labels[t][i])

                y_pred.append(pred_class)
                y_true.append(true_class)

                overall_conf.append(conf)
                label_conf[pred_class].append(conf)

                if pred_class == true_class:
                        count = count + 1
                        correct_overall_conf.append(conf)
                        correct_label_conf[pred_class].append(conf)
                else:
                        wrong_overall_conf.append(conf)
                        wrong_label_conf[pred_class].append(conf)

        pickle.dump(y_pred, open("y_pred"+str(t)+".p", "wb"))
        pickle.dump(y_true, open("y_true"+str(t)+".p", "wb"))
        pickle.dump(overall_conf, open("overall_conf"+str(t)+".p", "wb"))
        pickle.dump(correct_overall_conf, open("correct_overall_conf"+str(t)+".p", "wb"))
        pickle.dump(wrong_overall_conf, open("wrong_overall_conf"+str(t)+".p", "wb"))
        pickle.dump(label_conf, open("label_conf"+str(t)+".p", "wb"))
        pickle.dump(correct_label_conf, open("correct_label_conf"+str(t)+".p", "wb"))
        pickle.dump(wrong_label_conf, open("wrong_label_conf"+str(t)+".p", "wb"))
        acc = count/len(labels[t])
        acc_list.append(acc)
        print("fold: {}, no. of samples: {}, acc: {}".format(t, len(labels[t]), acc))

#print("All folds individual accuracies:", acc_list)
print("Average accuracy:", sum(acc_list)/len(acc_list))
