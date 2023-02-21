#!/usr/bin/env python
# coding: utf-8



import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from skimage.util.montage import montage2d
import os
import h5py
make_proj = lambda x: np.sum(x,1)[::-1]
make_mip = lambda x: np.max(x,1)[::-1]




get_ipython().run_line_magic('matplotlib', 'inline')
with h5py.File(os.path.join('..', 'input', 'lab_petct_vox_5.00mm.h5'), 'r') as p_data:
    id_list = np.random.permutation(list(p_data['ct_data'].keys()))
    print(list(p_data.keys()))
    ct_image = p_data['ct_data'][id_list[0]].value
    pet_image = p_data['pet_data'][id_list[0]].value
    label_image = (p_data['label_data'][id_list[0]].value>0).astype(np.uint8)
fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize = (12, 4))
ct_proj = make_proj(ct_image)
suv_max = make_mip(pet_image)
lab_proj = make_proj(label_image)
ax1.imshow(ct_proj, cmap = 'bone')
ax1.set_title('CT Image')
ax2.imshow(np.sqrt(suv_max), cmap = 'magma')
ax2.set_title('SUV Image')
ax3.imshow(lab_proj, cmap = 'gist_earth')
ax3.set_title('Tumor Labels')




pet_weight = 5.0 # how strongly to weight the pet_signal (1.0 is the same as CT)
petct_vol = np.stack([np.stack([(ct_slice+-200).clip(0,2048)/2048, 
                            pet_weight*(suv_slice).clip(0,5)/5.0
                           ],-1) for ct_slice, suv_slice in zip(ct_image, pet_image)],0)




get_ipython().run_cell_magic('time', '', 'from skimage.segmentation import slic\nfrom skimage.segmentation import mark_boundaries\ndef make_sp_seg(seed_val):\n    np.random.seed(seed_val)\n    return slic(petct_vol, \n                  n_segments = 10000, \n                  compactness = 0.1,\n                 multichannel = True)\npetct_segs = make_sp_seg(0)')




petct_max_segs = make_mip(petct_segs)
ct_proj = make_proj(petct_vol[:,:,:,0])
suv_mip = make_mip(petct_vol[:,:,:,1])

fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize = (14, 6))
ax1.imshow(suv_mip, cmap = 'magma')
ax1.set_title('SUV Image')
ax2.imshow(petct_max_segs, cmap = plt.cm.rainbow)
ax2.set_title('Segmented Image')
ax3.imshow(mark_boundaries(suv_mip, petct_max_segs))




for idx in np.unique(petct_segs):
    cur_region_mask = petct_segs == idx
    labels_in_region = label_image[cur_region_mask]
    labeled_regions_inside = np.unique(labels_in_region)
    if len(labeled_regions_inside)>1:
        print('Superpixel id', idx, 'regions', len(labeled_regions_inside))
        print('\n',pd.value_counts(labels_in_region))
        print('Missclassified Pixels:', np.sum(pd.value_counts(labels_in_region)[1:].values))




nz_labels = [i for i in np.unique(label_image) if i>=0]
fig, m_axs = plt.subplots(len(nz_labels), 2, figsize = (5, 15))
for (ax1, ax2), i_label in zip(m_axs, nz_labels):
    out_sp = np.zeros_like(petct_segs)
    cur_label_mask = label_image == i_label
    labels_in_region = petct_segs[cur_label_mask]
    
    superpixels_in_region = np.unique(labels_in_region)
    for i, sp_idx in enumerate(superpixels_in_region):
        out_sp[petct_segs == sp_idx] = i+1
    
    ax1.imshow(make_proj(cur_label_mask), cmap = 'bone')
    ax1.set_title('Label Map {}'.format(i_label) if i_label>0 else 'Background Label')
    ax1.axis('off')
    
    ax2.imshow(make_proj(out_sp), cmap = 'gist_earth')
    ax2.set_title('Superpixels ({})'.format(len(superpixels_in_region)))
    ax2.axis('off')




for idx in np.unique(label_image):
    cur_region_mask = label_image == idx
    labels_in_region = petct_segs[cur_region_mask]
    labeled_regions_inside = np.unique(labels_in_region)
    print('Label id', idx, 'superpixels inside', len(labeled_regions_inside))
    #print(pd.value_counts(labels_in_region))




from skimage.measure import regionprops
import warnings
from warnings import warn
def scalar_attributes_list(im_props):
    """
    Makes list of all scalar, non-dunder, non-hidden
    attributes of skimage.measure.regionprops object
    """

    attributes_list = []

    for i, test_attribute in enumerate(dir(im_props[0])):

        # Attribute should not start with _ and cannot return an array
        # does not yet return tuples
        try:
            if test_attribute[:1] != '_' and not                     isinstance(getattr(im_props[0], test_attribute), np.ndarray):
                attributes_list += [test_attribute]
        except Exception as e:
            warn("Not implemented: {} - {}".format(test_attribute, e), RuntimeWarning)

    return attributes_list


def regionprops_to_df(im_props):
    """
    Read content of all attributes for every item in a list
    output by skimage.measure.regionprops
    """

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        attributes_list = scalar_attributes_list(im_props)

    # Initialise list of lists for parsed data
    parsed_data = []

    # Put data from im_props into list of lists
    for i, _ in enumerate(im_props):
        parsed_data += [[]]

        for j in range(len(attributes_list)):
            parsed_data[i] += [getattr(im_props[i], attributes_list[j])]

    # Return as a Pandas DataFrame
    return pd.DataFrame(parsed_data, columns=attributes_list)




get_ipython().run_cell_magic('time', '', "sp_rprops = regionprops(petct_segs, intensity_image=pet_image)\nsp_rprop_df = regionprops_to_df(sp_rprops)\nprint('Region Analysis for ', len(sp_rprops), 'superpixels')")




# add a malignancy score
sp_rprop_df['malignancy'] = sp_rprop_df['label'].map(lambda sp_idx: np.mean(label_image[petct_segs==sp_idx]))
# add the mean CT value
sp_rprop_df['meanCT'] = sp_rprop_df['label'].map(lambda sp_idx: np.mean(petct_vol[:,:,:,0][petct_segs==sp_idx]))
sp_rprop_df.sample(3)




get_ipython().run_cell_magic('time', '', "out_df_list = [sp_rprop_df]\nfor seed_val in range(1,5):\n    t_petct_segs = make_sp_seg(seed_val)\n    sp_rprops = regionprops(t_petct_segs, intensity_image=pet_image)\n    sp_rprop_df = regionprops_to_df(sp_rprops)\n    # add a malignancy score\n    sp_rprop_df['malignancy'] = sp_rprop_df['label'].map(lambda sp_idx: np.mean(label_image[t_petct_segs==sp_idx]))\n    # add the mean CT value\n    sp_rprop_df['meanCT'] = sp_rprop_df['label'].map(lambda sp_idx: np.mean(petct_vol[:,:,:,0][t_petct_segs==sp_idx]))\n    out_df_list += [sp_rprop_df]\nsp_rprop_df = pd.concat(out_df_list)")




reg_var = 'malignancy'
# boost the malignancy count by 1e3
boost_df = sp_rprop_df.sample(10000, weights=(1e-3+sp_rprop_df[reg_var].values), replace = True)

# break into variables and outcomes
numeric_df = boost_df.select_dtypes(include=[np.number])
x_data = numeric_df[[ccol for ccol in numeric_df.columns if ccol not in [reg_var]]]
y_data = boost_df[reg_var].values




# predict the malignancy based on the other features
from sklearn.tree import DecisionTreeRegressor
malig_tree = DecisionTreeRegressor()
malig_tree.fit(x_data, y_data)




# show the accuracy (on the original data, which is clearly cheating)
y_predict = malig_tree.predict(x_data)
fig, ax1 = plt.subplots(1,1)
ax1.plot(y_data, y_predict, 'b+', label = '')
ax1.plot([0,1], [0,1], 'r-', label = 'ideal')
ax1.set_xlabel('Actual Malignancy')
ax1.set_ylabel('Predicted Malignancy')
ax1.legend()




from sklearn.tree import export_graphviz
from subprocess import check_call
from IPython.display import Image
def visualize_tree(tree, feature_names):
    """Create tree png using graphviz.

    Args
    ----
    tree -- scikit-learn DecsisionTree.
    feature_names -- list of feature names.
    """
    with open("dt.dot", 'w') as f:
        export_graphviz(tree, out_file=f,
                        feature_names=feature_names)

    command = ["dot", "-Tpng", "dt.dot", "-o", "dt.png"]
    try:
        check_call(command)
        return Image('dt.png')
    except Exception as e:
        raise RuntimeError("Could not run dot, ie graphviz, to "
             "produce visualization: {}".format(e))




visualize_tree(malig_tree, x_data.columns)




def get_code(tree, feature_names, target_names,
             spacer_base="    "):
    """Produce psuedo-code for decision tree.

    Args
    ----
    tree -- scikit-leant DescisionTree.
    feature_names -- list of feature names.
    target_names -- list of target (class) names.
    spacer_base -- used for spacing code (default: "    ").

    Notes
    -----
    based on http://stackoverflow.com/a/30104792.
    """
    left      = tree.tree_.children_left
    right     = tree.tree_.children_right
    threshold = tree.tree_.threshold
    features  = [feature_names[i] for i in tree.tree_.feature]
    value = tree.tree_.value

    def recurse(left, right, threshold, features, node, depth):
        spacer = spacer_base * depth
        if (threshold[node] != -2):
            print(spacer + "if ( " + features[node] + " <= " +                   str(threshold[node]) + " ) {")
            if left[node] != -1:
                    recurse(left, right, threshold, features,
                            left[node], depth+1)
            print(spacer + "}\n" + spacer +"else {")
            if right[node] != -1:
                    recurse(left, right, threshold, features,
                            right[node], depth+1)
            print(spacer + "}")
        else:
            target = value[node]
            for i, v in zip(np.nonzero(target)[1],
                            target[np.nonzero(target)]):
                target_name = target_names[i]
                target_count = int(v)
                print(spacer + "return " + str(target_name) +                       " ( " + str(target_count) + " examples )")

    recurse(left, right, threshold, features, 0, 0)




# meant for decision trees but shows we have made something
get_code(malig_tree, x_data.columns, ['regression_value'])




get_ipython().run_cell_magic('time', '', "n_petct_segs = make_sp_seg(0)\nnsp_rprops = regionprops(n_petct_segs, intensity_image=pet_image)\nnsp_rprop_df = regionprops_to_df(sp_rprops)\nnsp_rprop_df['meanCT'] = nsp_rprop_df['label'].map(lambda sp_idx: np.mean(petct_vol[:,:,:,0][n_petct_segs==sp_idx]))\n\nfnsp_rprop_df=nsp_rprop_df.select_dtypes(include=[np.number])\nfnsp_rprop_df=fnsp_rprop_df[[ccol for ccol in numeric_df.columns if ccol not in [reg_var]]]\nnsp_rprop_df['score']=malig_tree.predict(fnsp_rprop_df)")




get_ipython().run_cell_magic('time', '', "n_img=np.zeros(n_petct_segs.shape,dtype=np.float32)\nfor _, n_row in nsp_rprop_df.iterrows():\n    n_img[n_petct_segs==n_row['label']]=n_row['score']")




get_ipython().run_line_magic('matplotlib', 'inline')
overlay_image=0.5*plt.cm.bone(petct_vol[:,:,:,0])+0.5*plt.cm.inferno(n_img)
fig, m_axes = plt.subplots(3,3, figsize = (12, 12))
for c_title,c_img,(ax1, ax2, ax3) in zip(['Prediction','Label','Overlay'],
                           [n_img,label_image,overlay_image],
                           m_axes):
    
    for i, (cax, clabel) in enumerate(zip([ax1, ax2, ax3], ['xy', 'zy', 'zx'])):
        cax.imshow(np.max(c_img[::-1],i).squeeze(), interpolation='none', cmap = 'bone_r',vmin=0,vmax=1)
        cax.set_title('%s %s Projection' % (c_title,clabel))
        cax.set_xlabel(clabel[0])
        cax.set_ylabel(clabel[1])
        cax.axis('off')




get_ipython().run_line_magic('matplotlib', 'inline')
pos_img=(label_image==1)*n_img
neg_img=(label_image==0)*(n_img)

overlay_image=plt.cm.bone(petct_vol[:,:,:,0])

fig, m_axes = plt.subplots(3,3, figsize = (12, 12))
for c_title,c_img,(ax1, ax2, ax3) in zip(['CT Image','True Positives','False Positives'],
                           [petct_vol[:,:,:,0],pos_img,neg_img],
                           m_axes):
    
    for i, (cax, clabel) in enumerate(zip([ax1, ax2, ax3], ['xy', 'zy', 'zx'])):
        cax.imshow(np.mean(c_img[::-1],i).squeeze(), interpolation='none', cmap = 'bone')
        cax.set_title('%s %s Projection' % (c_title,clabel))
        cax.set_xlabel(clabel[0])
        cax.set_ylabel(clabel[1])
        cax.axis('off')

