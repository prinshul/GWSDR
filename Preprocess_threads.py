
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
from cv2 import DualTVL1OpticalFlow_create as DualTVL1
from tensorflow.python.platform import app, flags
import os
import sys
import cv2
import threading



import tensorflow as tf
import numpy as np


DATA_DIR = '/DATA/kin600/kin600_20/'
SAVE_DIR = '/DATA/kin600/kin600_20_flow/'

_EXT = ['.avi', '.mp4']
extension = '.mp4'
_IMAGE_SIZE = 224
_CLASS_NAMES = '/DATA/keras-kinetics-i3d/c.txt'

FLAGS = flags.FLAGS

flags.DEFINE_string('data_dir', DATA_DIR, 'directory containing data.')
flags.DEFINE_string('save_to', SAVE_DIR, 'where to save flow data.')
flags.DEFINE_string('name', 'Autism', 'dataset name.')
flags.DEFINE_integer('num_threads',90, 'number of threads.')


def _video_length(video_path):
    _, ext = os.path.splitext(video_path)
    if not ext in _EXT:
        raise ValueError('Extension "%s" not supported' % ext)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Could not open the file.\n{}".format(video_path))
    if cv2.__version__ >= '3.0.0':
        CAP_PROP_FRAME_COUNT = cv2.CAP_PROP_FRAME_COUNT
    else:
        CAP_PROP_FRAME_COUNT = cv2.cv.CV_CAP_PROP_FRAME_COUNT
    length = int(cap.get(CAP_PROP_FRAME_COUNT))
    return length

def compute_rgb(video_path):
    cap = cv2.VideoCapture(video_path)
    rgb = []
    vid_len = _video_length(video_path)
    for _ in range(vid_len):
        ret, frame2 = cap.read()
        if(ret==False):
                continue
        curr = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
        curr = cv2.resize(curr, (_IMAGE_SIZE, _IMAGE_SIZE))
        curr = np.array(curr,dtype = np.float64)
        curr = 2*(curr - np.min(curr))/np.ptp(curr)-1
        rgb.append(curr)
    cap.release()
    rgb = np.array(rgb, dtype = np.float64)
    return rgb

def compute_TVL1(video_path):
    print(video_path)
    TVL1 = DualTVL1()
    cap = cv2.VideoCapture(video_path)

    ret, frame1 = cap.read()
    if(ret==False):
        return
    prev = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    prev = cv2.resize(prev, (_IMAGE_SIZE, _IMAGE_SIZE))
    flow = []
    vid_len = _video_length(video_path)
    for _ in range(vid_len-1):
        ret, frame2 = cap.read()
        if(ret==False):
                continue
        curr = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        curr = cv2.resize(curr, (_IMAGE_SIZE, _IMAGE_SIZE))
        curr_flow = TVL1.calc(prev, curr, None)
        assert(curr_flow.dtype == np.float32)
    # truncate [-20, 20]
        curr_flow[curr_flow >= 20] = 20
        curr_flow[curr_flow <= -20] = -20
    # scale to [-1, 1]
        #max_val = lambda x: max(max(x.flatten()), abs(min(x.flatten())))
        #curr_flow = curr_flow / max_val(curr_flow)
        curr_flow = 2*(curr_flow - np.min(curr_flow))/np.ptp(curr_flow)-1
        flow.append(curr_flow)
        prev = curr
    cap.release()
    flow = np.array(flow)
    return flow

def _process_video_files(thread_index, filenames, save_to):
    for filename in filenames:
        #flow = compute_rgb(filename)
        flow = compute_TVL1(filename)
        fullname, _ = os.path.splitext(filename)
        split_name = fullname.split('/')
        save_name = os.path.join(save_to, split_name[-2], split_name[-1] + '.npy')
        np.save(save_name, flow)
        print("%s [thread %d]: %s done." % (datetime.now(), thread_index, filename))
        sys.stdout.flush()

def _process_dataset():

    filenames = []
    for subdir, dirs, files in os.walk(DATA_DIR):
        for file in files:
            if(file=='.DS_Store' or file.endswith('backup')):
                continue
            path= os.path.join(subdir, file)
            filenames.append(path)

#     filenames = [filename
#                for class_fold in
#                  tf.gfile.Glob(os.path.join(FLAGS.data_dir,'*'))
#                  for filename in
#                    tf.gfile.Glob(os.path.join(class_fold, '*'))
#               ]
    #print(filenames)
    filename_chunk = np.array_split(filenames, FLAGS.num_threads)
    threads = []

  # Create a mechanism for monitoring when all threads are finished.
    coord = tf.train.Coordinator()

  # Launch a thread for each batch.
    print("Launching %s threads." %  FLAGS.num_threads)
    for thread_index in range(FLAGS.num_threads):
        args = (thread_index, filename_chunk[thread_index], FLAGS.save_to)
        t = threading.Thread(target=_process_video_files, args=args)
        t.start()
        threads.append(t)

  # Wait for all the threads to terminate.
    coord.join(threads)
    print("%s: Finished processing all %d videos in data set '%s'." %
        (datetime.now(), len(filenames), FLAGS.name))


def main(unused_argv):
    if not tf.gfile.IsDirectory(FLAGS.save_to):
        tf.gfile.MakeDirs(FLAGS.save_to)
        f = open(_CLASS_NAMES)
        classes = [cls.strip() for cls in f.readlines()]
        for cls in classes:
            tf.gfile.MakeDirs(os.path.join(FLAGS.save_to, cls))

    _process_dataset()


if __name__ == '__main__':
    app.run()


# In[2]:


get_ipython().run_line_magic('tb', '')
