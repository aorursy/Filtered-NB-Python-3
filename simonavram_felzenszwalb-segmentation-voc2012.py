#!/usr/bin/env python
# coding: utf-8



import os
import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt

import scipy.misc
import skimage.segmentation
import skimage.feature

from tqdm import tqdm
import imageio

from copy import copy




def image_segmentation(img, scale = 1.0, sigma = 0.8, min_size = 50):
        
        img_normalize = skimage.util.img_as_float(img) # Convert an image to floating point format, with values in [0, 1].
        im_mask = skimage.segmentation.felzenszwalb(
                        img_normalize, 
                        scale = scale, 
                        sigma = sigma,
                        min_size = min_size)
        img_seg = np.dstack([img,im_mask])
        
        return(img_seg) #output: shape = (height, width, 4) 4-L label of region




img_dir = os.listdir('../input/pascal-voc-2012/VOC2012/JPEGImages/')
Number_of_plots = 1

for imgnumer in (img_dir[0:Number_of_plots]):

    img = imageio.imread(os.path.join('../input/pascal-voc-2012/VOC2012/JPEGImages/',imgnumer)) #read image
    img_seg = image_segmentation(img = img, min_size = 1000) #get segmentation
    
    #Show plot
    figure = plt.figure(figsize=(20,40))
    ax = figure.add_subplot(1,2,1)
    ax.imshow(img)
    ax.set_title("original")
    ax = figure.add_subplot(1,2,2)
    ax.imshow(img_seg[:,:,3])
    ax.set_title("Felzenszwalb’s efficient graph based segmentation".format(len(np.unique(img_seg[:,:,3]))))
    plt.show()




def get_regions_dictionaray(img):
    
    #input img : (height, width, 4) 4 - number of chanels [R,G,B,L] L - label
    #only L channel use in this method
    
    #output is dictionaray Regions
    #{0: {'labels': [0], 'max_x': 131, 'max_y': 74, 'min_x': 0,   'min_y': 0},
    # 1: {'labels': [1], 'max_x': 189, 'max_y': 37, 'min_x': 75,  'min_y': 0}
    
    
    # get segmentation output
    img_seg = img[:,:,3]
    Regions = {}
    for y, i in enumerate(img_seg): # get y axis 

        for x, l in enumerate(i): # get x axis
            # add new region
            if l not in Regions:
                Regions[l] = {"min_x": np.Inf, 
                        "min_y": np.Inf,
                        "max_x": 0, 
                        "max_y": 0, 
                        "labels": [l]}

            # bounding box
            if Regions[l]["min_x"] > x:
                Regions[l]["min_x"] = x
            if Regions[l]["min_y"] > y:
                Regions[l]["min_y"] = y
            if Regions[l]["max_x"] < x:
                Regions[l]["max_x"] = x
            if Regions[l]["max_y"] < y:
                Regions[l]["max_y"] = y
    ## drop region if x<0 or y<0
    
    Region_copy = copy(Regions)
    for key in Regions.keys():
        r = Regions[key]
        if (r["min_x"] == r["max_x"]) or (r["min_y"] == r["max_y"]):
            del Region_copy[key]
    return(Region_copy)

Regions = get_regions_dictionaray(img_seg)
print(f"Regions was found: {len(Regions)}")




def plt_rectangle(plt,label,x1,y1,x2,y2,color = "yellow", alpha=0.5):
    linewidth = 3
    if type(label) == list:
        linewidth = len(label)*3 + 2
        label = ""
        
    plt.text(x1,y1,label,fontsize=20,backgroundcolor=color,alpha=alpha)
    plt.plot([x1,x1],[y1,y2], linewidth=linewidth,color=color, alpha=alpha)
    plt.plot([x2,x2],[y1,y2], linewidth=linewidth,color=color, alpha=alpha)
    plt.plot([x1,x2],[y1,y1], linewidth=linewidth,color=color, alpha=alpha)
    plt.plot([x1,x2],[y2,y2], linewidth=linewidth,color=color, alpha=alpha)

#Show original image
figure = plt.figure(figsize=(20,40))
ax = figure.add_subplot(1,2,1)
ax.imshow(img_seg[:,:,:3]/2**8)
ax.set_title("original")

#Original image add lables
for item, color in zip(Regions.values(),sns.xkcd_rgb.values()):#use seaborn color palette
    x1 = item["min_x"]
    y1 = item["min_y"]
    x2 = item["max_x"]
    y2 = item["max_y"]
    label = item["labels"][0]
    plt_rectangle(plt,label,x1,y1,x2,y2,color=color)

#Show segmentet image
ax = figure.add_subplot(1,2,2)
ax.imshow(img_seg[:,:,3])
ax.set_title("Felzenszwalb’s efficient graph based segmentation".format(len(np.unique(img_seg[:,:,3]))))

#Segmentation image add lables 
for item, color in zip(Regions.values(),sns.xkcd_rgb.values()):#use seaborn color palette
    x1 = item["min_x"]
    y1 = item["min_y"]
    x2 = item["max_x"]
    y2 = item["max_y"]
    label = item["labels"][0]
    plt_rectangle(plt,label,x1,y1,x2,y2,color=color)
plt.show()




def get_texture_gradient(img):
    """
        calculate texture gradient for entire image

        output will be [height(*)][width(*)]
    """
    ret = np.zeros(img.shape[:3])
    for colour_channel in (0, 1, 2):
        ret[:, :, colour_channel] = skimage.feature.local_binary_pattern(
            img[:, :, colour_channel], 50, 1.0)#8

    return ret

def plot_image_with_min_max(img,nm):
    img = img[:,:,:3]
    plt.figure(figsize=(10,10))
    plt.imshow(img)
    plt.title("{} min={:5.3f}, max={:5.3f}".format(nm,
                                                   np.min(img),
                                                   np.max(img)))
    plt.show()

tex_grad = get_texture_gradient(img)   
plot_image_with_min_max(tex_grad,nm="tex_grad")




def calc_hsv(img):
    hsv = skimage.color.rgb2hsv(img[:,:,:3])
    return(hsv)
hsv = calc_hsv(img)
plot_image_with_min_max(hsv,nm="hsv")




def get_hist(img, minhist=0, maxhist=1):
   
    BINS = 25
    hist = np.array([])

    for colour_channel in range(3): # For rgb

        # get one color channel 
        c = img[:, colour_channel]

        # calculate histogram for each colour and join to the result
        hist = np.concatenate(
            [hist] + [np.histogram(c, BINS, 
                                   # The lower and upper range of the bins. 
                                   (minhist, maxhist))[0]])

    # L1 normalize
    hist = hist / len(img)
    return hist

def get_regions_with_histogram_info(tex_grad, img_seg, Regions,hsv,tex_trad):#img
    
    for k, v in list(Regions.items()):

        masked_pixels  = hsv[img_seg[:, :, 3] == k] 
        Regions[k]["size"]   = len(masked_pixels / 4)
        Regions[k]["hist_c"] = get_hist(masked_pixels,minhist=0, maxhist=1)

        # texture histogram
        Regions[k]["hist_t"] = get_hist(tex_grad[img_seg[:, :, 3] == k],minhist=0, maxhist=2**8-1)
    return(Regions)

Regions = get_regions_with_histogram_info(tex_grad, img_seg,Regions,hsv,tex_grad)




def extract_neighbours(regions):

    def intersectiom_of_regions(a, b):#check if two regions intersect
        if (a["min_x"] < b["min_x"] < a["max_x"] and a["min_y"] < b["min_y"] < a["max_y"]) or           (a["min_x"] < b["max_x"] < a["max_x"] and a["min_y"] < b["max_y"] < a["max_y"]) or           (a["min_x"] < b["min_x"] < a["max_x"] and a["min_y"] < b["max_y"] < a["max_y"]) or           (a["min_x"] < b["max_x"] < a["max_x"] and a["min_y"] < b["min_y"] < a["max_y"]):
            return True
        return False

    Regions = list(regions.items())
    neighbours = []
    for cur, a in enumerate(Regions[:-1]):
        for b in Regions[cur + 1:]:
            if intersectiom_of_regions(a[1], b[1]):
                neighbours.append((a, b))

    return neighbours

neighbours = extract_neighbours(Regions)
print(f"Out of {len(Regions)} regions, was found {len(neighbours)} neighbours")




#calculate the sum of histogram intersection of colour
def simularity_of_colour(r1, r2): 
    return sum([min(a, b) for a, b in zip(r1["hist_c"], r2["hist_c"])])

#calculate the sum of histogram intersection of texture
def simularity_of_texture(r1, r2):
    return sum([min(a, b) for a, b in zip(r1["hist_t"], r2["hist_t"])])

#calculate the size similarity over the image
def simularity_of_size(r1, r2, imsize):
    return 1.0 - (r1["size"] + r2["size"]) / imsize

#calculate the fill similarity over the image
def simularity_of_fill(r1, r2, imsize):
    bbsize = (
        (max(r1["max_x"], r2["max_x"]) - min(r1["min_x"], r2["min_x"]))
        * (max(r1["max_y"], r2["max_y"]) - min(r1["min_y"], r2["min_y"]))
    )
    return 1.0 - (bbsize - r1["size"] - r2["size"]) / imsize

#return sum of simularity of two regions
def calculate_sum_sim(r1, r2, imsize):
    return (simularity_of_colour(r1, r2)       +            simularity_of_texture(r1, r2)      +            simularity_of_size(r1, r2, imsize) +            simularity_of_fill(r1, r2, imsize))

#calculate simularity of all regions
def calculate_similarlity(img,neighbours,verbose=False):
    imsize = img.shape[0] * img.shape[1]
    Simularity = {}
    for (ai, ar), (bi, br) in neighbours:
        Simularity[(ai, bi)] = calculate_sum_sim(ar, br, imsize)
        if verbose:
            print("S[({:2.0f}, {:2.0f})]={:3.2f}".format(ai,bi,Simularity[(ai, bi)]))
    return(Simularity)

print("S[(Pair of the intersecting regions)] = Similarity index")
Simularity = calculate_similarlity(img,neighbours,verbose=True)




#Merge two regions base on compare of x1 and x2
def merge_regions(r1, r2):
    
    new_size = r1["size"] + r2["size"]
    rt = {
        "min_x": min(r1["min_x"], r2["min_x"]),
        "min_y": min(r1["min_y"], r2["min_y"]),
        "max_x": max(r1["max_x"], r2["max_x"]),
        "max_y": max(r1["max_y"], r2["max_y"]),
        "size": new_size,
        "hist_c": (r1["hist_c"] * r1["size"] + r2["hist_c"] * r2["size"]) / new_size,
        "hist_t": (r1["hist_t"] * r1["size"] + r2["hist_t"] * r2["size"]) / new_size,
        "labels": r1["labels"] + r2["labels"]
    }
    return rt

def merge_regions_in_order(Simularity,Regions,imsize, verbose=False):
 
    # hierarchal search
    while Simularity != {}:
        
        #Sort dictionary by simularity 
        i, j = sorted(Simularity.items(), key=lambda i: i[1])[-1][0] 

        #Marge the regions and add to the region dictionary
        t = max(Regions.keys()) + 1.0
        Regions[t] = merge_regions(Regions[i], Regions[j])

       
        #remove all the pair of regions where one of the regions was selected in sort 
        key_to_delete = []
        for k, v in list(Simularity.items()):
            if (i in k) or (j in k):
                key_to_delete.append(k)
        for k in key_to_delete:
            del Simularity[k]

        #calculate similarity of new merged region and the regions and its intersecting region (intersecting region is the region that are to be deleted)
        for k in key_to_delete:
            if k != (i,j):
                if k[0] in (i, j):
                    n = k[1]
                else:
                    n = k[0]
                Simularity[(t, n)] = calculate_sum_sim(Regions[t], Regions[n], imsize)
    if verbose:
        print("{} regions".format(len(Regions)))

    # return list of regions
    regions = []
    for k, r in list(Regions.items()):
            regions.append({
                'rect': (
                    r['min_x'],              # min x
                    r['min_y'],              # min y
                    r['max_x'] - r['min_x'], # width 
                    r['max_y'] - r['min_y']),# height
                'size': r['size'],
                'labels': r['labels']
            })
    return(regions)



regions = merge_regions_in_order(Simularity,Regions,img.shape[0]*img.shape[1],verbose=True)




plt.figure(figsize=(20,20))    
plt.imshow(img_seg[:,:,:3]/2**8)
for item, color in zip(regions,sns.xkcd_rgb.values()):
    x1, y1, width, height = item["rect"]
    label = item["labels"][:5]
    plt_rectangle(plt,label,x1,y1,x2 = x1 + width,y2 = y1 + height, color = color)
plt.show()

plt.figure(figsize=(20,20))    
plt.imshow(img_seg[:,:,3])
for item, color in zip(regions,sns.xkcd_rgb.values()):
    x1, y1, width, height = item["rect"]
    label = item["labels"][:5]
    plt_rectangle(plt,label,
                  x1,
                  y1,
                  x2 = x1 + width,
                  y2 = y1 + height, color= color)
plt.show()




def get_region_proposal(img_original,min_size = 500):
    img        = image_segmentation(img_original,min_size = min_size)#Get regions by Felzenszwalb’s algorithm 
    Regions    = get_regions_dictionaray(img)#Labled regions to dictionary 
    tex_grad   = get_texture_gradient(img)#get texture gradient
    hsv        = calc_hsv(img)#get hsv image reprezentaiton
    Regions    = get_regions_with_histogram_info(tex_grad, img, Regions,hsv,tex_grad)#get histogram info

    del tex_grad, hsv
    
    neighbours = extract_neighbours(Regions)#get list of neighbours regions
    Sim        = calculate_similarlity(img,neighbours)#get simularity of heighbours
    regions    = merge_regions_in_order(Sim,Regions,imsize = img.shape[0] * img.shape[1])#merge regions to new 
    return(regions)

img  = imageio.imread('../input/pascal-voc-2012/VOC2012/JPEGImages/2007_000039.jpg')
regions = get_region_proposal(img,min_size=1000)

regions
print("{} regions are found".format(len(regions)))

#Plot
plt.figure(figsize=(20,20))    
plt.imshow(img[:,:,:3]/2**8)
for item, color in zip(regions,sns.xkcd_rgb.values()):
    x1, y1, width, height = item["rect"]
    label = item["labels"][:5]
    plt_rectangle(plt,label,x1,y1,x2 = x1 + width,y2 = y1 + height, color = color)
plt.show()

