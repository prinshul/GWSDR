import pickle
import os
import numpy as np

src = '/DATA/kin600/kin_600_armax_all_data_200samples_flow/'
des = '/DATA/kin600/kin_600_armax_all_data_200samples_flow_2sec/'
fpsdic = pickle.load( open( "/DATA/keras-kinetics-i3d/fps.p", "rb" ) )
for fld in os.listdir(src):
    d = des + '/' + fld
    os.makedirs(d)
    s = src + '/' + fld
    for file in os.listdir(s):
        sfile = s + '/' + file
        a= np.load(sfile)
        key = file.split('.')[0]
        if not key in fpsdic:
                continue
                
        fps = fpsdic[key][0]
        fcount = fpsdic[key][1]
        cnt = round(2.0*fps)
        cnt = int(cnt)
        i=0
        tfcount = fcount
        j=0
        #tcnt = cnt
        tcnt=5
        td = d + '/' + key
        os.makedirs(td)
        while(tfcount > tcnt):
            b=a[i:cnt,:,:,:,]
            print(i)
            print(cnt)
            np.save(td+'/'+key+'_'+str(j)+'.npy',b)
            j=j+1
            i=i+5
            #i=cnt
            i=int(i)
            cnt = cnt+tcnt
            cnt = int(cnt)
            tfcount = tfcount - tcnt
            if(tfcount - tcnt < 10):
                break	
        tfcount = int(tfcount)
        b=a[i:i+tfcount,:,:,:,]
        print(i)
        print(i+tfcount)
        np.save(td+'/'+key+'_'+str(j)+'.npy',b)
