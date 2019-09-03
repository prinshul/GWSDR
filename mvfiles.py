import shutil
import os
c=0
import pickle
src='/DATA/kin600/kin_600_armax_all_data_200samples_flow_2sec/'
des='/DATA/kin600/kin_600_armax_all_data_200samples_flow_2sec_all/'

traindata = pickle.load( open( "/DATA/keras-kinetics-i3d/2s_mm_all.p", "rb" ) )
#traindata= os.listdir(src)
for file in traindata:
        fname=os.path.basename(file)
        dir1 = os.path.dirname(os.path.normpath(file))
        dir2 = os.path.dirname(os.path.normpath(dir1))
        d=os.path.basename(dir2)
        print(d)
        if not os.path.exists(des+d):
                os.makedirs(des+d)
        c=0
        #for file in os.listdir(src+d):
        shutil.copy(file,des+d+'/'+fname)
                #c=c+1
                #if(c==200):
                  #      break
                

