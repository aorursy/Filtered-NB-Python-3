#!/usr/bin/env python
# coding: utf-8



https://github.com/fastai/course-v3/blob/master/nbs/dl1/lesson2-download.ipynb




from fastai.vision import *




folder = 'black'
file = 'urls_black.txt'




folder = 'teddys'
file = 'urls_teddys.txt'




folder = 'grizzly'
file = 'urls_grizzly.txt'




work_p = Path("./")
path = Path('../input')
dest = work_p/folder
dest.mkdir(parents=True, exist_ok=True)




classes = ['teddys','grizzly','black']




download_images(path/file, dest, max_pics=200)




# If you have problems download, try with `max_workers=0` to see exceptions:
# download_images(path/file, dest, max_pics=20, max_workers=0)




for c in classes:
    print(c)
    verify_images(work_p/c, delete=True, max_size=500)




np.random.seed(2)
data = ImageDataBunch.from_folder(work_p, train=".", valid_pct=0.2,
        ds_tfms=get_transforms(), size=224, num_workers=2, bs=16).normalize(imagenet_stats)




# If you already cleaned your data, run this cell instead of the one before
# np.random.seed(42)
# data = ImageDataBunch.from_csv(".", folder=".", valid_pct=0.2, csv_labels='cleaned.csv',
#         ds_tfms=get_transforms(), size=224, num_workers=4).normalize(imagenet_stats)




data.classes




data.show_batch(rows=3, figsize=(7,8))




data.classes, data.c, len(data.train_ds), len(data.valid_ds)




learn = create_cnn(data, models.resnet34, metrics=error_rate)




learn.fit_one_cycle(4)




learn.save('stage-1')




learn.unfreeze()




learn.lr_find()




learn.recorder.plot()




learn.fit_one_cycle(2, max_lr=slice(3e-5,1e-4))




learn.save('stage-2')




learn.load('stage-2');




interp = ClassificationInterpretation.from_learner(learn)




interp.plot_confusion_matrix()




from fastai.widgets import *




ds, idxs = DatasetFormatter().from_toplosses(learn, ds_type=DatasetType.Valid)




ImageCleaner(ds, idxs, work_p)




ds, idxs = DatasetFormatter().from_similars(learn, ds_type=DatasetType.Valid)




ImageCleaner(ds, idxs, work_p, duplicates=True)




learn.export()




defaults.device = torch.device('cpu')




img = open_image(work_p/'black'/'00000021.jpg')
img




learn = load_learner(work_p)




pred_class,pred_idx,outputs = learn.predict(img)
pred_class




learn = create_cnn(data, models.resnet34, metrics=error_rate)




learn.fit_one_cycle(1, max_lr=0.5)




learn = create_cnn(data, models.resnet34, metrics=error_rate)




learn.fit_one_cycle(5, max_lr=1e-5)




learn.recorder.plot_losses()




learn = create_cnn(data, models.resnet34, metrics=error_rate, pretrained=False)




learn.fit_one_cycle(1)




work_p




get_ipython().system('rm core')




get_ipython().system('ls -l')




np.random.seed(2)
data = ImageDataBunch.from_folder(work_p, train=".", valid_pct=0.9, bs=16, 
        ds_tfms=get_transforms(do_flip=False, max_rotate=0, max_zoom=1, max_lighting=0, max_warp=0
                              ),size=224, num_workers=4).normalize(imagenet_stats)




learn = create_cnn(data, models.resnet50, metrics=error_rate, ps=0, wd=0)
learn.unfreeze()




learn.fit_one_cycle(40, slice(1e-6,1e-4))






