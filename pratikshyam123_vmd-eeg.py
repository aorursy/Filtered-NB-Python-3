#!/usr/bin/env python
# coding: utf-8



get_ipython().system('pip install vmdpy')




import matplotlib.pyplot as plt
from vmdpy import VMD  
import os
from scipy import signal
from scipy import interpolate
from scipy import signal
import numpy as np
import math
import cmath
import pandas as pd
import pickle
import time 
from os import listdir
from os.path import isfile,isdir, join




def create_directory(name):
    try:
        #Create target Directory
        os.mkdir(name)
        print("Directory ",name,"Created")
    except FileExistsError:
        print("Directory",name,"already exists")
    return name

def save_fig(x,y,color,title,dir_):
    plt.figure() #figsize=(10,20)
    plt.plot(x,y,color)
    plt.title(title)
    plt.savefig(dir_)
    plt.close()




def calc_area_of_sodp(X,Y,i,channel):
    #Area of Second Order Difference Plot
    SX = math.sqrt(np.sum(np.multiply(X,X))/len(X))
    SY = math.sqrt(np.sum(np.multiply(Y,Y))/len(Y))
    SXY = np.sum(np.multiply(X,Y))/len(X)
    D = cmath.sqrt((SX*SX) + (SY*SY) - (4*(SX*SX*SY*SY - SXY*SXY)))
    a = 1.7321 *cmath.sqrt(SX*SX + SY*SY + D)
    b = 1.7321 * cmath.sqrt(SX*SX + SY*SY - D)
    Area = math.pi *a *b
    print("Channel=  ",channel,"Area of SODP of IMF number= ",i, " is ", Area)
    return Area

def calc_mean_and_ctm(X,Y,i,channel):
    features = pd.DataFrame(columns=['radius','mean_distance','central_tendency_measure'])
    r = 0.5
    d = [ math.sqrt(X[i]*X[i] + Y[i]*Y[i]) for i in range (0,len(X))]
    delta = [1 if i<r else 0 for i in d]
    d = [i for i in d if i<r]
        
    ctm = np.sum(delta[:-2])/(len(delta)-2)
    mean_distance = np.mean(d)
    
    features.loc[0] = [r] + [ctm] + [mean_distance]
    return features




def second_order_difference_plot(y, i, channel,dir_,imp_features,trial):
    #remove outliers
    upper_quartile = np.percentile(y,80)
    lower_quartile = np.percentile(y,20)
    IQR = (upper_quartile - lower_quartile) * 1.5
    quartileSet = (lower_quartile- IQR, upper_quartile +IQR)
    y = y[np.where((y >= quartileSet[0]) & (y <= quartileSet[1]))]
    
    #plotting SODP
    X = np.subtract(y[1:],y[0:-1]) #x(n+1)-x(n)
    Y = np.subtract(y[2:],y[0:-2]).tolist()#x(n+2)-x(n-1)
    Y.extend([0])
    #save_fig(X,Y,'.','SODP'+str(i),dir_+'/SODP'+str(i)+'.png')
    
    Area = calc_area_of_sodp(X,Y,i,channel)
    features =calc_mean_and_ctm(X,Y,i,channel)
    
    df = pd.DataFrame({"Trial":trial,"Channel":channel,"SODP_No":i,"Area":Area,
                       "m(r=0.5)":features[features['radius']==0.5]['mean_distance'],
                       "ctm(r=0.5)":features[features['radius']==0.5]['central_tendency_measure']})
    imp_features = imp_features.append(df,ignore_index=True)
    return imp_features




def vmd(channel, s, dirc_,imp_features,trial):
    T = len(s)  
    fs = 1/T  
    t = np.arange(1,T+1)/T 
    freqs = 2*np.pi*(t-0.5-fs)/(fs)  
    
    if(T > 40000):
        ll = 180
    else:
        ll = 60
    x1 = np.linspace(0,ll,T)

    f = s
    f_hat = np.fft.fftshift((np.fft.fft(f)))

    #. some sample parameters for VMD  
    alpha = 2000       # moderate bandwidth constraint  
    tau = 0.            # noise-tolerance (no strict fidelity enforcement)  
    K = 8              # 3 modes  
    DC = 0             # no DC part imposed  
    init = 1           # initialize omegas uniformly  
    tol = 1e-7  

    #. Run actual VMD code  
    u, u_hat, omega = VMD(f, alpha, tau, K, DC, init, tol)  

    N = u.T.shape[1]

    #plt.figure(figsize=(20,40))
    #plt.subplot(N+1,1,1)
    #plt.plot(x1, s, 'r')
    #plt.title("Input signal")
    #plt.xlabel("Time [s]")

    for n in range (0,N+1):
        #try:
            #plt.subplot(N+1,1,n+2)
        #except:
            #break
        #plt.plot(x1, u.T[:,n], 'g')
        #plt.title("Mode "+str(n+1))
        #plt.xlabel("Time [s]")
        try:
            imp_features = second_order_difference_plot(u.T[:,n],str(n+1),channel,dirc_,imp_features,trial)
        except:
            break


    #plt.tight_layout()
    #plt.savefig('sub_'+ subject+'_'+ trial+'_'+channel+'_VMD.png')
    #plt.savefig(dirc_+'/'+str(channel)+'.png')
    #plt.close()
    return imp_features




path_src = "/kaggle/input/vmd111/"

onlyfiles = [d for d in listdir(path_src) if isfile(join(path_src, d))] 
print(onlyfiles)





for filename in onlyfiles:
    imp_features = pd.DataFrame(columns=['Trial','Channel','SODP_No','Area','m(r=0.5)','ctm(r=0.5)'])
    subject_no = int(filename[7:9])
    trial_no = int(filename[10:11])
    file = pd.read_excel(path_src+'/'+filename)
    if 'Unnamed: 0' in file.columns:
            file.drop(columns=['Unnamed: 0'], inplace= True)
    file.reset_index(drop = True,inplace = True)
    channels = file.columns
    #t = np.linspace(0,60,len(file.index))
    
    #curr_dir = create_directory(str(subject_no))
    #dir_  = create_directory(curr_dir +'/Trial '+ str(trial_no))
    #print("Currently in ", dir_," directory.")
    
    for channel in channels:
        print(subject_no ,"   ",trial_no,"  ",channel)
        s = file[channel].values
        imp_features = vmd(channel,s, "dir_",imp_features,trial_no)
        #writer = pd.ExcelWriter(dir_+'/Trial'+str(trial_no)+'.xlsx')
        #imp_features.to_excel(writer, index=False)
        #writer.save()
        imp_features.to_excel('/kaggle/working/'+str(subject_no)+'_Trial'+str(trial_no)+'_'+channel+'.xlsx')

