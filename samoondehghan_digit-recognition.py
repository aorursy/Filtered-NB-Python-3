#!/usr/bin/env python
# coding: utf-8



import pandas as pd
from fastai.vision import *




#train = pd.read_csv('train.csv')




#train.head()




#y_train = train['label']




#x_train = train.iloc[:,1:].to_numpy() 




#plt.imshow(x_train[3].reshape((28,28)), cmap="gray")




# x_train,y_train = map(torch.tensor, (x_train,y_train))
# n,c = x_train.shape
# x_train.shape, y_train.min(), y_train.max()




class MyDataset(TensorDataset):
    # "Sample numpy array dataset"
    def __init__(self, x, y):
        self.x, self.y = x, y
        self.c = 10
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, i):
        return self.x[i], self.y[i]




from sklearn.model_selection import train_test_split

def createTensorDataSet(path):
  df = pd.read_csv(path)
  y = df['label'].to_numpy()
  x = df.iloc[:,1:].to_numpy()
  x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.03,random_state=42)
  x_train,y_train,x_valid,y_valid = map(torch.tensor, (x_train,y_train,x_valid,y_valid))
  train_ds = MyDataset(x_train, y_train)
  valid_ds = MyDataset(x_valid, y_valid)
  tfms = get_transforms(do_flip=False)
  data = ImageDataBunch.create(train_ds, valid_ds, bs=64)
  return data




#data = createTensorDataSet('train.csv')
tfms = get_transforms(do_flip=False)
df = pd.read_csv('train.csv')
data = ImageDataBunch.from_df('/root/', df,  size=26)




# "data.show_batch(rows=3, figsize=(5,5))




doc(ImageList.from_folder)




class PixelImageItemList(ImageList):
    def open(self,fn):
        regex = re.compile(r'\d+')
        fn = re.findall(regex,fn)
        df = self.inner_df[self.inner_df.fn.values == int(fn[0])]
        df_fn = df[df.fn.values == int(fn[0])]
        img_pixel = df_fn.drop(labels=['label','fn'],axis=1).values
        img_pixel = img_pixel.reshape(28,28)
        img_pixel = np.stack((img_pixel,)*3,axis=-1)
        return vision.Image(pil2tensor(img_pixel,np.float32).div_(255))




df_train = pd.read_csv('train.csv')
df_train['fn'] = df_train.index
df_train.head()




src = (PixelImageItemList.from_df(df_train,'.',cols='fn')
      .split_by_rand_pct()
      .label_from_df(cols='label'))




data = (src.transform(tfms=(rand_pad(padding=5,size=28,mode='zeros'),[]))
       .databunch(bs=64)
       .normalize(imagenet_stats))




data.show_batch(rows=3, figsize=(5,5))




learn = cnn_learner(data, models.resnet34, metrics=accuracy)




learn.fit_one_cycle(2)




learn.lr_find()
learn.recorder.plot()




learn.unfreeze()




learn.fit_one_cycle(3, max_lr=slice(1e-6,1e-4))




class CustomImageItemList(ImageList):
    def open(self, fn):
        img = fn.reshape(28,28)
        img = np.stack((img,)*3, axis=-1) # convert to 3 channels
        return Image(pil2tensor(img, dtype=np.float32))

    @classmethod
    def from_csv_custom(cls, path:PathOrStr, csv_name:str, imgIdx:int=1, header:str='infer', **kwargs)->'ItemList':
        df = pd.read_csv(Path(path)/csv_name, header=header)
        res = super().from_df(df, path=path, cols=0, **kwargs)
        # convert pixels to an ndarray
        res.items = df.iloc[:,imgIdx:].apply(lambda x: x.values / 255.0, axis=1).values
        return res




# note: there are no labels in a test set, so we set the imgIdx to begin at the 0 col
test = CustomImageItemList.from_csv_custom(path='.', csv_name='test.csv', imgIdx=0)




tfms = get_transforms(do_flip=False)
data = (CustomImageItemList.from_csv_custom(path='.', csv_name='train.csv')
                           .split_by_rand_pct(.2)
                           .label_from_df(cols='label')
                           .add_test(test, label=0)
                           .transform(tfms)
                           .databunch(bs=64)
                           .normalize(imagenet_stats))




data.show_batch(rows=3, figsize=(5,5))




learn = cnn_learner(data, models.resnet34, metrics=accuracy)




learn.fit_one_cycle(4)




learn.lr_find()
learn.recorder.plot()




learn.unfreeze()




learn.fit_one_cycle(3, max_lr=slice(1e-4,1e-3))




predictions, *_ = learn.get_preds(DatasetType.Test)
labels = np.argmax(predictions, 1)
# output to a file
submission_df = pd.DataFrame({'ImageId': list(range(1,len(labels)+1)), 'Label': labels})
submission_df.to_csv(f'submission.csv', index=False)




submission_df.head()




from google.colab import files
files.download('submission.csv') 




get_ipython().system('ls')









import pandas as pd
sample_submission = pd.read_csv("../input/digit-recognizer/sample_submission.csv")
test = pd.read_csv("../input/digit-recognizer/test.csv")
train = pd.read_csv("../input/digit-recognizer/train.csv")

