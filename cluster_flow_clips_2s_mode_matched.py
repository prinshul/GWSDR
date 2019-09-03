import os
import pickle
import numpy as np
from keras.models import load_model
from i3d_inception import Inception_Inflated3d

LockHands=['playing trombone']
# 'breathing fire',
# 'tasting beer',
# 'eating burger']

Touchear=['playing ice hockey']
# 'jaywalking',
# 'eating chips',
# 'tasting wine']

Touchhead=['skydiving']
# 'sword swallowing',
# 'swinging baseball bat',
# 'combing hair']

Touchnose=['putting on eyeliner']
# 'javelin throw',
# 'diving cliff',
# 'ice climbing']

Move_the_table=['riding a bike']
# 'sanding floor',
# 'pushing cart',
# 'riding camel']

Rolly_Polly=['playing hand clapping games']
# 'washing hands',
# 'shuffling cards',
# 'making balloon shapes']

Arms_Up=['cartwheeling']
# 'spinning poi',
# 'roller skating',
# 'standing on hands']

Tapping=['playing drums']
# 'playing organ',
# 'tapping guitar',
# 'playing xylophone']


#LockHands=['playing harp',
#'using inhaler']
# 'inflating balloons',
# 'arguing',
# 'decorating the christmas tree']

#Touchear=['sucking lolly',
#'sled dog racing']
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
#data_to_exclude= pickle.load( open( "../crossval5_8/aug_200samples_f_70_train7_20e.p", "rb" ) ) 
#s =set()
#for f in data_to_exclude:
#    if 'autism_flow' in f: 
#        continue
#    s.add(os.path.basename(f)) 
flow_model = Inception_Inflated3d(include_top=False, weights='flow_imagenet_and_kinetics', input_shape=(None, 224, 224, 2), classes=8)
flow_model.load_weights("/DATA/keras-kinetics-i3d/data/0_8/n_flow_wts_7.h5")
#flow_model.load_weights("/DATA/keras-kinetics-i3d/data/0_8/weighted_154_70_200samples_above_20e0.h5")
flow_model.summary()
flow_map = {}
    
kin_classes =[Move_the_table,Touchear,LockHands,Touchhead,Touchnose,Rolly_Polly,Tapping, Arms_Up]
c=0
i=0
filelistkins=[]
for kinlist in kin_classes:
    cj=0
    for kclass in kinlist:
        f = os.listdir('/DATA/kin600/kin_600_armax_all_data_200samples_flow_2sec/'+kclass)
        for folder in f:
            files = os.listdir('/DATA/kin600/kin_600_armax_all_data_200samples_flow_2sec/'+kclass+'/'+folder)
            fdict={}
            for file in files:
                #if(file in s):
                #	continue
                tx = '/DATA/kin600/kin_600_armax_all_data_200samples_flow_2sec/'+kclass+'/'+folder+'/' +file
                x = np.load(tx)
                #print(kinlist)   
                x = x.reshape((1, x.shape[0], x.shape[1], x.shape[2], x.shape[3]))
                flow_logits = flow_model.predict(x)
                flow_logits = flow_logits[0]
                sample_predictions = np.exp(flow_logits) / np.sum(np.exp(flow_logits))
                cls=np.argmax(sample_predictions)
                if(i!=cls):
                        continue
                #print(sample_predictions)
                fdict[tx]=sample_predictions[i]
                #fdict[os.path.basename(tx)]=sample_predictions[i]
            sorted_by_value = sorted(fdict.items(), key=lambda kv: kv[1],reverse=True)
            #print(sorted_by_value)
            #if(sorted_by_value[0][1]>=0.5):
            if sorted_by_value:
                c=c+1
                cj=cj+1
                filelistkins.append(sorted_by_value[0][0])

    i=i+1  
    print(kinlist,cj)
print('count:',c)
pickle.dump(filelistkins, open("2s_mm_all.p", "wb"))
