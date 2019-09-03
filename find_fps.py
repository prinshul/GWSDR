#LockHands=['getting a tattoo',
#'making jewelry',
#'bottling',
#'picking fruit',
#'mushroom foraging']

#Touchear=['tying necktie',
#'tying bow tie',
#'massaging neck',
#'playing trombone',
#'playing trumpet',
#'playing ocarina']


#Touchhead=["shaving head",
#"combing hair",
#"blowdrying hair",
#"massaging person's head",
#"washing hair",
#"dyeing hair",
#"curling hair",
#"ironing hair"]

#Touchnose=[
#'eating chips',
#'putting on lipstick',
#'blowing nose',
#'brushing teeth',
#'putting on foundation',
#'smoking pipe']

#Move_the_table=['riding snow blower',
#'sanding floor',
#'mopping floor',
#'pushing car',
#'pushing wheelbarrow']

#Rolly_Polly=['playing hand clapping games',
#'juggling balls',
#'juggling fire',
#'front raises',
#'clapping']

#Arms_Up=['clean and jerk',
#'standing on hands',
#'roller skating',
#'jumping jacks',
#'high jump']


#Tapping=['playing keyboard',
#'tapping guitar',
#'playing lute',
#'playing drums',
#'tapping pen']

Move_the_table=['riding a bike']

Touchear=['playing ice hockey']


LockHands=['playing trombone']


Touchhead=['skydiving']

Touchnose=['putting on eyeliner']

Rolly_Polly=['playing hand clapping games']

Tapping=['playing drums']


Arms_Up=['cartwheeling'] 
 
kin_classes =[Move_the_table,Touchear,LockHands,Touchhead,Touchnose,Rolly_Polly,Tapping, Arms_Up]

import pickle
import os
import numpy as np
from random import shuffle
import cv2
#traindata = pickle.load( open( "/DATA/crossval5_8/f_train0.p", "rb" ) )
#labeldata = pickle.load( open( "/DATA/crossval5_8/l_train0.p", "rb" ) )
traindata={}
labeldata=[]

i=0
for folder in kin_classes:
    for fld in folder:
        for file in os.listdir("/DATA/kin600/kin_600_armax_all_data_200samples/"+fld):
            file= file.split('.')[0]+'.mp4'
            x = "/DATA/kin600/kin_600_armax_all_data_200samples/"+fld+'/'+file
            cap = cv2.VideoCapture(x)
            current_framerate = cap.get(5)
            nframes = cap.get(7)
            x = os.path.basename(x).split('.')[0] 
            traindata[x]= [current_framerate,nframes]
            cap.release()
        

pickle.dump(traindata, open("/DATA/keras-kinetics-i3d/fps.p", "wb"))
print("data created ")
