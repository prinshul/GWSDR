 
Move_the_table=['driving tractor']
  
Touchear=['passing soccer ball']
LockHands=['playing harp']
Touchhead=['spinning poi']
Touchnose=['eating ice cream']
Rolly_Polly=['juggling fire']
Tapping=['playing maracas']
Arms_Up=['snatch weight lifting'] 
#LockHands=['playing laser tag',
#'playing trombone']
# 'breathing fire',
# 'tasting beer',
# 'eating burger']
#LockHands=['playing trombone']
#Touchear=['playing ice hockey',
#'blowing bubble gum']
# 'jaywalking',
# 'eating chips',
# 'tasting wine']

#Touchear=['playing ice hockey']
#Touchhead=[]
#Touchhead=['skydiving']
# 'head stand']
# 'sword swallowing',
# 'swinging baseball bat',
# 'combing hair']
#Touchnose=[]
#Touchnose=['putting on eyeliner']
#Touchnose=['disc golfing',
#'putting on eyeliner']
# 'javelin throw',
# 'diving cliff',
# 'ice climbing']
#Move_the_table=[]
#Move_the_table=['riding a bike']
#Move_the_table=['making the bed',
#'riding a bike']
# 'sanding floor',
# 'pushing cart',
# 'riding camel']
#Rolly_Polly=[]
#Rolly_Polly=['playing hand clapping games']
#Rolly_Polly=['playing hand clapping games',
#'playing rubiks cube']
# 'washing hands',
# 'shuffling cards',
# 'making balloon shapes']

#Arms_Up=[]
#Arms_Up=['cartwheeling']
#'cartwheeling']
# 'spinning poi',
# 'roller skating',
# 'standing on hands']
#Tapping=[]
#Tapping=['playing drums']
#Tapping=['playing drums',
#'dribbling basketball']
# 'playing organ',
# 'tapping guitar',
# 'playing xylophone']


# LockHands=['playing harp',
# 'using inhaler',
# 'inflating balloons',
# 'arguing',
# 'decorating the christmas tree']

# Touchear=['sucking lolly',
# 'sled dog racing',
# 'bowling',
# 'eating carrots',
# 'parasailing']

# Touchhead=['blowdrying hair',
# 'using a paint roller',
# 'using a sledge hammer',
# 'stretching leg',
# 'fixing hair']

# Touchnose=['flying kite',
# 'trimming trees',
# 'historical reenactment',
# 'putting on mascara',
# 'auctioning']

# Move_the_table=['falling off bike',
# 'jumping bicycle',
# 'pushing wheelbarrow',
# 'smelling feet',
# 'crawling baby']

# Rolly_Polly=['playing maracas',
# 'punching person (boxing)',
# 'punching bag',
# 'pillow fight',
# 'cracking knuckles']

# Arms_Up=['climbing a rope',
# 'dunking basketball',
# 'pull ups',
# 'front raises',
# 'swinging on something']

# Tapping=['flipping pancake',
# 'playing keyboard',
# 'playing bass guitar',
# 'shucking oysters',
# 'dancing charleston']
c_size=[191,192,117,147,154,121,141,183] 
kin_classes =[Move_the_table,Touchear,LockHands,Touchhead,Touchnose,Rolly_Polly,Tapping, Arms_Up]
import random
import pickle
import os
import numpy as np
from random import shuffle
import random

traindata = pickle.load( open( "/DATA/crossval5_8/5050_f_train0.p", "rb" ) )
labeldata = pickle.load( open( "/DATA/crossval5_8/5050_l_train0.p", "rb" ) )
c=0
i=0
kins_traindata=[]
kins_labeldata=[]
n = 3
for folder in kin_classes:
    t=0
    flg=0
    for fld in folder:
        if(flg==1):
                break
        if not os.path.exists("/DATA/kin600/kin600_200samples_argmax_5050_flow_2secs/"+fld):
                continue
        files=os.listdir("/DATA/kin600/kin600_200samples_argmax_5050_flow_2secs/"+fld)
        ls=range(0,len(files))
        indx=random.sample(ls,c_size[kin_classes.index(folder)])
        for j in indx:
        #for file in os.listdir("/DATA/kin600/kin_600_armax_all_data_200samples_flow/"+fld):
        #files = os.listdir("/DATA/kin600/kin600_2sec_flow_agn_80/"+fld)   
        #ix = np.random.randint(0,len(files),n)
        #for mx in range(0,n):
            x = "/DATA/kin600/kin600_200samples_argmax_5050_flow_2secs/"+fld+'/'+files[j]
            traindata.append(x)
            kins_traindata.append(x)
            y = np.zeros((1, 8))
            y[0,i]=1
            t=t+1
            c=c+1
            labeldata.append(y)
            kins_labeldata.append(y)
            if(t==20):
                flg=1
                break
            #print(x)
    i=i+1
    
flow_shuf_train = []
labels_shuf_train = []
index_shuf = [i for i in range(len(labeldata))]
shuffle(index_shuf)
flow_shuf_train = [traindata[i] for i in index_shuf]
#shuffle(index_shuf)
labels_shuf_train = [labeldata[i] for i in index_shuf]
        
#flow_shuf_train=np.asarray(flow_shuf_train)
#labels_shuf_train = np.asarray(labels_shuf_train)
print(c)
pickle.dump(labels_shuf_train, open("../crossval5_8/aug_l_mm_all_5050_alle"+".p", "wb"))
pickle.dump(flow_shuf_train, open("../crossval5_8/aug_f_mm_all_5050_alle"+".p", "wb"))
pickle.dump(kins_traindata, open("../crossval5_8/f_kins_mm_all_5050_alle"+".p", "wb"))
pickle.dump(kins_labeldata, open("../crossval5_8/l_kins_mm_all_5050_alle"+".p", "wb"))
#pickle.dump(labeldata, open("../crossval5_8/aug_l_all0"+".p", "wb"))
#pickle.dump(traindata, open("../crossval5_8/aug_f_all0"+".p", "wb"))
print("data created ")
