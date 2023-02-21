#!/usr/bin/env python
# coding: utf-8



import numpy as np
import pandas
import matplotlib.pyplot as plt
import sklearn
import seaborn as sns
import scipy




path_of_csv="../input/s1-an-1-/s1_an_1.csv"
sns.set()




def get_landmark(path):
    

    data=pandas.read_csv(path_of_csv)
    #print(data.head())

    l1=[]
    l2=[]
    p=[]
    for j in range(data.shape[0]):
        for i in range(68):
            s=" x_"+str(i)
            s1=" y_"+str(i)
            
                
            l1=[data[s][j],data[s1][j]]
            l2.append(l1)
            #lx.append(data[s][0])
            #ly.append(data[s1][0])
        p.append(l2)
        l2=[]
    #print(data.columns.values)
    #print(p)
    print(len(p))
    print(len(p[0]))
    print(len(p[0][0]))
    print(p[0][0][1])

    return p
#print(ly)




p=get_landmark(path_of_csv) 
p1=np.array(p)
print(p1)
print(p1.shape)




def plot_landmark(frame,frame_no=-1):
    
    for i in range(len(frame)):
        #print(i)
        x=frame[i][0]
        y=frame[i][1] #***Figure out why flipped in y-axis necessitating this -(minus) sign***
        #plt.text(x,y,i,ha="center", va="center",fontsize=8)
        plt.scatter(x,y,c='r', s=10)
        plt.title('Landmark Points frame #'+str(i))
        
        # Set x-axis label
        plt.xlabel('X')
        # Set y-axis label
        plt.ylabel('Y')
    ax=plt.gca()
    ax.invert_yaxis()    
    plt.show()




def plot_landmark_on_frame(frame,path_of_image):  
    
    im=plt.imread(path_of_image)
    implot = plt.imshow(im)
    #plt.show()
    
    for i in range(len(frame)):
        #print(i)
        x=frame[i][0]
        y=frame[i][1]
        plt.text(x,y,i,ha="center", va="center",fontsize=8)
        plt.scatter(x,y,c='r', s=40)
        plt.title(f'Landmark Points frame #{str(i)}')
        # Set x-axis label
        plt.xlabel('X')
        # Set y-axis label
        plt.ylabel('Y')
    plt.show()   




for i in range(50):
        plot_landmark(p[i])
        path="../input/frames/frame"+str(i)+".jpg"
        plot_landmark_on_frame(p[i],path)




from sklearn.cluster import KMeans
import numpy as np

no_of_clusters=3
#X = np.array([[1, 2], [2, 3]], [1, 4, 2], [1, 0, 2], [4, 2, 3], [4, 4, 8], [4, 0, 6]])
print(p1.shape[0])
print(p1[0].shape)
new = np.reshape(p1,(p1.shape[0],68*2))
print(new.shape)
print(new)
print(p1[:,:,0])
print(p1[:,:,0].shape)

kmeans = KMeans(n_clusters=no_of_clusters, random_state=0).fit(new)

kmeans_y = KMeans(n_clusters=no_of_clusters, random_state=0).fit(p1[:,:,1])
kmeans_x = KMeans(n_clusters=no_of_clusters, random_state=0).fit(p1[:,:,0])





Nc = range(1, 20)

kmeas = [KMeans(n_clusters=i) for i in Nc]

kmeans
Y = new
score = [kmeas[i].fit(Y).inertia_ for i in range(len(kmeas))]

score

plt.plot(Nc,score)

plt.xlabel('Number of Clusters')

plt.ylabel('Score')

plt.title('Elbow Curve')

plt.show()




print(kmeans.labels_)
print(kmeans_x.labels_)
print(kmeans_y.labels_)




centrr = kmeans.cluster_centers_
centroid_x=kmeans_x.cluster_centers_
centroid_y=kmeans_y.cluster_centers_
print(centrr.shape)
cen = np.reshape(centrr,(no_of_clusters,68,2))
#print(centrr)
print(cen.shape)
print(cen)




l=[]
l11=[]
centroid=[]
for i in range(centroid_x.shape[0]):
    for j in range(centroid_x.shape[1]):
        l=[centroid_x[i,j],centroid_y[i,j]]
        #print(l)
        l11.append(l)
        #print(l11)
        #print("\n\n")
    centroid.append(l11)
    l11=[]

cent=np.array(centroid)
print(cent.shape)
print(cent)









import scipy
#z = scipy.spatial.distance.cdist(A,B,'chebyshev')
minframe=[]
#print(z)
for i in range(no_of_clusters):

    dist=[]
    for j in range(50):
        y = scipy.spatial.distance.cdist(p1[j],cen[i],'euclidean')#cent
        #print(y)
        #print(np.sum(np.diag(y)))
        dist.append(np.trace(y))
        #print(str(i)+"  "+str(np.trace(y)))
    dd=np.array(dist)
    minframe.append(np.argmin(dd))
print(minframe)
    




list_of_images=[]
for i in minframe:
    path_of_image="../input/frames/frame"+str(i)+".jpg"
    im=plt.imread(path_of_image)
    list_of_images.append(im)
    implot = plt.imshow(im)
    plt.show()




def grid_display(list_of_images, list_of_titles=[], no_of_columns=2, figsize=(10,10)):

    fig = plt.figure(figsize=figsize)
    column = 0
    for i in range(len(list_of_images)):
        column += 1
        #  check for end of column and create a new figure
        if column == no_of_columns+1:
            fig = plt.figure(figsize=figsize)
            column = 1
        fig.add_subplot(1, no_of_columns, column)
        plt.imshow(list_of_images[i])
        plt.axis('off')
        if len(list_of_titles) >= len(list_of_images):
            plt.title(list_of_titles[i])




grid_display(list_of_images, list_of_titles=[], no_of_columns=3, figsize=(10,10))




dist=[]
for i in range(0,len(arr),2):
    dist.append(((arr[i][0]-arr[i+1][0])**2 + (arr[i][1]-arr[i+1][1])**2)**0.5)

print(dist)




from math import acos
def d(x1,y1,x2,y2):
    val=((x1-x2)**2+(y1-y2)**2)**0.5
    return val
angle=[]
print(len(arr))
def 
    return (acos(((d(arr[i][0],arr[i][1],arr[i+1][0],arr[i+1][1])**2+
                        d(arr[i][0],arr[i][1],arr[i+2][0],arr[i+2][1])**2-
                        d(arr[i+1][0],arr[i+1][1],arr[i+2][0],arr[i+2][1])**2)/
    (2*d(arr[i][0],arr[i][1],arr[i+1][0],arr[i+1][1]
    *d(arr[i][0],arr[i][1],arr[i+2][0],arr[i+2][1])))
    )))
print(angle)
print(len(angle))

