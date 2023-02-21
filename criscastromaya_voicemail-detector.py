#!/usr/bin/env python
# coding: utf-8



get_ipython().system('pip install torch-lr-finder')




get_ipython().system('pip install torchsummary')




get_ipython().system('wget https://bin.equinox.io/c/4VmDzA7iaHb/ngrok-stable-linux-amd64.zip')
get_ipython().system('unzip ngrok-stable-linux-amd64.zip ')




import os
import numpy as np
import torch
import librosa
import torchaudio
import pandas as pd
import torch.nn as nn
import itertools as it, glob
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.distributed import get_rank
from torch.distributed import get_world_size
from torch.utils.data.sampler import Sampler
from torch.utils.data.sampler import BatchSampler




import itertools as it, glob
def multiple_patterns(*patterns):
    return it.chain.from_iterable(glob.iglob(pattern) for pattern in patterns)




root_path="/kaggle/input/testdata/"
mar_1_10="1-10mar_120-150secs-data_chunked_mostly_speech/1-10mar_120-150secs-data_chunked_mostly_speech/*.wav"
feb_1_14="1-14feb_120-150secs-data_chunked_mostly_speech/1-14feb_120-150secs-data_chunked_mostly_speech/*.wav"
df_m_speech = pd.DataFrame()
for i,filename in enumerate(multiple_patterns(f"{root_path}{feb_1_14}",f"{root_path}{mar_1_10}")):
    df_m_speech = df_m_speech.append({0:filename, 1: 'speech'}, ignore_index=True)




df_m_speech




utterances = {os.path.split(filename)[-1]: pd.read_csv(filename,header=None)  for filename in glob.glob("/kaggle/input/testdata/*.csv")}




for name,df in utterances.items():
    if df.loc[0,df.columns[0]] in {'speech','voicemail','silence','beep'}:
        utterances[name].loc[:]=df[[df.columns[1],df.columns[0]]]
    n_path=f"{name.split('labeled')[0]}data_chunked"
    utterances[name].loc[:,0]=utterances[name][0].apply(lambda p:os.path.join(root_path,f"{n_path}/{n_path}",p))




def mergeDataframes(**dataframes):
    return pd.concat(list(dataframes.values()),axis=0).reset_index(drop=True)
def getFrequencyDistribution(df,column_name):
    return df[pd.notnull(df[column_name])].groupby(df[column_name]).size()




utterances['mostly_speech']=df_m_speech




df=mergeDataframes(**utterances)




getFrequencyDistribution(df,1).plot.bar()




df=df[(df[1]!='silence') & (df[1]!='beep')].copy()
df.reset_index(inplace=True,drop=True)




df




getFrequencyDistribution(df,1).plot.bar()




def train_validate_test_split(df, train_percent=.70, validate_percent=.15, seed=None):
       np.random.seed(seed)
       perm = np.random.permutation(df.index)
       m = len(df.index)
       train_end = int(train_percent * m)
       validate_end = int(validate_percent * m) + train_end
       train = df.iloc[perm[:train_end]]
       validate = df.iloc[perm[train_end:validate_end]]
       test = df.iloc[perm[validate_end:]]
       return train, validate, test




train, val, test=train_validate_test_split(df, train_percent=.75, validate_percent=.23, seed=None)




def pdist(vectors):
    vectors=vectors.view(-1,vectors.size()[1])
    distance_matrix = -2 * vectors.mm(torch.t(vectors)) + vectors.pow(2).sum(dim=1).view(1, -1) + vectors.pow(2).sum(
        dim=1).view(-1, 1)
    return distance_matrix




def random_hard_negative(loss_values):
    hard_negatives = np.where(loss_values > 0)[0]
    return np.random.choice(hard_negatives) if len(hard_negatives) > 0 else None




class TripletSelector:
    """
    Implementation should return indices of anchors, positive and negative samples
    return np array of shape [N_triplets x 3]
    """

    def __init__(self):
        pass

    def get_triplets(self, embeddings, labels):
        raise NotImplementedError




from itertools import combinations
class FunctionNegativeTripletSelector(TripletSelector):
    """
    For each positive pair, takes the hardest negative sample (with the greatest triplet loss value) to create a triplet
    Margin should match the margin used in triplet loss.
    negative_selection_fn should take array of loss_values for a given anchor-positive pair and all negative samples
    and return a negative index for that pair
    """

    def __init__(self, margin, negative_selection_fn, cpu=True):
        super(FunctionNegativeTripletSelector, self).__init__()
        self.cpu = cpu
        self.margin = margin
        self.negative_selection_fn = negative_selection_fn

    def get_triplets(self, embeddings, labels):
        if self.cpu:
            embeddings = embeddings.cpu()
        distance_matrix = pdist(embeddings)
        distance_matrix = distance_matrix.cpu()

        labels = labels.cpu().data.numpy()
        triplets = []

        for label in set(labels):
            label_mask = (labels == label)
            label_indices = np.where(label_mask)[0]
            if len(label_indices) < 2:
                continue
            negative_indices = np.where(np.logical_not(label_mask))[0]
            anchor_positives = list(combinations(label_indices, 2))  # All anchor-positive pairs
            anchor_positives = np.array(anchor_positives)

            ap_distances = distance_matrix[anchor_positives[:, 0], anchor_positives[:, 1]]
            for anchor_positive, ap_distance in zip(anchor_positives, ap_distances):
                loss_values = ap_distance - distance_matrix[torch.LongTensor(np.array([anchor_positive[0]])), torch.LongTensor(negative_indices)] + self.margin
                loss_values = loss_values.data.cpu().numpy()
                hard_negative = self.negative_selection_fn(loss_values)
                if hard_negative is not None:
                    hard_negative = negative_indices[hard_negative]
                    triplets.append([anchor_positive[0], anchor_positive[1], hard_negative])

        if len(triplets) == 0:
            triplets.append([anchor_positive[0], anchor_positive[1], negative_indices[0]])

        triplets = np.array(triplets)

        return torch.LongTensor(triplets)




def RandomNegativeTripletSelector(margin, cpu=False): 
    return FunctionNegativeTripletSelector(margin=margin,
                                           negative_selection_fn=random_hard_negative,
                                           cpu=cpu)




class OnlineTripletLoss(nn.Module):
    """
    Online Triplets loss
    Takes a batch of embeddings and corresponding labels.
    Triplets are generated using triplet_selector object that take embeddings and targets and return indices of
    triplets
    """

    def __init__(self, margin, triplet_selector):
        super(OnlineTripletLoss, self).__init__()
        self.margin = margin
        self.triplet_selector = triplet_selector

    def forward(self, embeddings, target):

        triplets = self.triplet_selector.get_triplets(embeddings, target)

        if embeddings.is_cuda:
            triplets = triplets.cuda()

        ap_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 1]]).pow(2).sum(1)  # .pow(.5)
        an_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 2]]).pow(2).sum(1)  # .pow(.5)
        losses = F.relu(ap_distances - an_distances + self.margin)

        return losses.mean(), len(triplets)




class AverageNonzeroTripletsMetric():
    '''
    Counts average number of nonzero triplets found in minibatches
    '''

    def __init__(self):
        self.values = 0
        self.steps = 0

    def __call__(self, outputs, target, loss):
        self.values+=loss[1]
        self.steps+=1
        return self.value()

    def reset(self):
        self.values = 0
        self.steps = 0

    def value(self):
        return self.values/self.steps

    def name(self):
        return 'Average nonzero triplets'




class SpectrogramParser():    
    def __init__(self, audio_conf, normalize=False, speed_volume_perturb=False, spec_augment=False,mono=True,filterbanks=64):
        """
        Parses audio file into spectrogram with optional normalization and various augmentations
        :param audio_conf: Dictionary containing the sample rate, window and the window length/stride in seconds
        :param normalize(default False):  Apply standard mean and deviation normalization to audio tensor
        :param speed_volume_perturb(default False): Apply random tempo and gain perturbations
        :param spec_augment(default False): Apply simple spectral augmentation to mel spectograms
        """
        super(SpectrogramParser, self).__init__()
        self.windows = {'hamming': torch.hamming_window, 'hann': torch.hann_window, 'blackman': torch.blackman_window,
           'bartlett': torch.bartlett_window}
        self.window_stride = audio_conf['window_stride']
        self.window_size = audio_conf['window_size']
        self.sample_rate = audio_conf['sample_rate']
        self.window = self.windows.get(audio_conf['window'], None)
        self.mono=mono
        self.normalize = normalize
        self.n_fft=int(self.sample_rate * self.window_size)
        self.hop_length=int(self.sample_rate * self.window_stride)
        self.filterbanks=filterbanks
        self.spectransformer=torchaudio.transforms.MelSpectrogram(n_fft=self.n_fft,
                                                                  hop_length=self.hop_length,
                                                                  win_length=self.n_fft,
                                                                  n_mels=self.filterbanks,
                                                                  sample_rate=self.sample_rate,
                                                                  window_fn=self.window)
        self.amptodb=torchaudio.transforms.AmplitudeToDB(stype='power', top_db=80)
        
    def parse_audio(self, audio_path):
        y,sr = librosa.load(audio_path,self.sample_rate,mono=self.mono)
        y=torch.FloatTensor(y)
        mel = self.spectransformer(y)
        spect = self.amptodb(mel)
        if self.normalize:
            mean = spect.mean()
            std = spect.std()
            spect.add_(-mean)
            spect.div_(std)
        return spect




class SpectrogramDataset(Dataset, SpectrogramParser):
    def __init__(self, audio_conf,dataframe, labels, normalize=False, speed_volume_perturb=False, spec_augment=False):
        """
        Dataset that loads tensors via a csv containing file paths to audio files and transcripts separated by
        a comma. Each new line is a different sample. Example below:
        /path/to/audio.wav,label
        ...
        :param audio_conf: Dictionary containing the sample rate, window and the window length/stride in seconds
        :param manifest_filepath: Path to manifest csv as describe above
        :param labels: String containing all the possible characters to map to
        :param normalize: Apply standard mean and deviation normalization to audio tensor
        :param speed_volume_perturb(default False): Apply random tempo and gain perturbations
        :param spec_augment(default False): Apply simple spectral augmentation to mel spectograms
        """
        ids=dataframe.values
        self.ids = ids
        self.size = len(ids)
        self.labels_set=set(labels)
        self.targets=ids[:,1]
        self.labels_map = dict([(labels[i], i) for i in range(len(labels))])
        super(SpectrogramDataset, self).__init__(audio_conf, normalize, speed_volume_perturb, spec_augment)

    def __getitem__(self, index):
        sample = self.ids[index]
        audio_path, audio_label = sample[0], sample[1]
        spect = self.parse_audio(audio_path)
        audio_label=self.labels_map.get(audio_label,None)
        return spect, audio_label

    def __len__(self):
        return self.size




def _collate_fn(batch):
    def func(p):
        return p[0].size(1)
    freq_size,max_seqlength = max(batch, key=func)[0].size()
    minibatch_size = len(batch)
    inputs = torch.zeros(minibatch_size, 1, freq_size, max_seqlength)
    targets = []
    for x in range(minibatch_size):
        sample = batch[x]
        tensor = sample[0]
        target = sample[1]
        seq_length = tensor.size(1)
        inputs[x][0].narrow(1, 0, seq_length).copy_(tensor)
        targets.append(target)
    targets = torch.tensor(targets)
    return inputs, targets 


class AudioDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        """
        Creates a data loader for AudioDatasets.
        """
        super(AudioDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = _collate_fn




class BalancedBatchSampler(BatchSampler):
    """
    BatchSampler - from a MNIST-like dataset, samples n_classes and within these classes samples n_samples.
    Returns batches of size n_classes * n_samples
    """

    def __init__(self, labels, n_classes, n_samples):
        self.labels = labels
        self.labels_set = list(set(self.labels))
        self.label_to_indices = {label: np.where(self.labels == label)[0]
                                 for label in self.labels_set}
        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.n_dataset = len(self.labels)
        self.batch_size = self.n_samples * self.n_classes

    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size < self.n_dataset:
            try:
                classes = np.random.choice(self.labels_set, self.n_classes, replace=False) #orgiinal
            except:
                print(self.labels_set," ",self.n_classes)
                raise "terrible error" 
            indices = []
            for class_ in classes:
                indices.extend(self.label_to_indices[class_][
                               self.used_label_indices_count[class_]:self.used_label_indices_count[
                                                                         class_] + self.n_samples])
                self.used_label_indices_count[class_] += self.n_samples
                if self.used_label_indices_count[class_] + self.n_samples > len(self.label_to_indices[class_]):
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0
            yield indices
            self.count += self.n_classes * self.n_samples

    def __len__(self):
        return self.n_dataset // self.batch_size




audio_conf = dict(sample_rate=16000,
                          window_size=.025,
                          window_stride=.01,
                          window="hamming")
test_dataset = SpectrogramDataset(audio_conf,test, labels=["voicemail","speech"],
                                       normalize=True)




test_batch_sampler = BalancedBatchSampler(test_dataset.targets, n_classes=2, n_samples=4)
test_loader = AudioDataLoader(test_dataset,batch_sampler=test_batch_sampler)




iterator_loader=iter(test_loader)
import time
start_time = time.time()
sample=next(iterator_loader)
print("--- %s seconds to generate a minibatch ---" % (time.time() - start_time))
specs,labels=sample




import librosa.display
import matplotlib
import matplotlib.pyplot as plt
def visualization_spectrogram(mel_spectrogram, title):
    """visualizing result of SpecAugment
    # Arguments:
      mel_spectrogram(ndarray): mel_spectrogram to visualize.
      title(String): plot figure's title
    """
    # Show mel-spectrogram using librosa's specshow.
    plt.figure(figsize=(10, 4))
    #mel_spec_db = librosa.power_to_db(mel_spectrogram[0].numpy(), ref=np.max)
    mel_spec_db=mel_spectrogram[0].numpy()
    librosa.display.specshow(mel_spec_db, x_axis='time', y_axis='mel')
    plt.colorbar()
    # plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.tight_layout()
    plt.show()




specs,labels=sample
reverse_map=dict(map(reversed, test_dataset.labels_map.items()))
for spec,label in zip(*sample):
    visualization_spectrogram(spec,reverse_map.get(label.item()))




from torch.utils.tensorboard import SummaryWriter
def fit(train_loader, val_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda, log_interval, metrics=[],exp_prefix='',
        start_epoch=0):
    """
    Loaders, model, loss function and metrics should work together for a given task,
    i.e. The model should be able to process data output of loaders,
    loss function should process target output of loaders and outputs from the model
    Examples: Classification: batch loader, classification model, NLL loss, accuracy metric
    Online triplet learning: batch loader, embedding model, online triplet loss
    """
    TRAIN_LOG='Epoch: {}/{}. Train set: Average loss: {:.4f}'
    METRIC_LOG='\t{}: {}'
    VAL_LOG='\nEpoch: {}/{}. Validation set: Average loss: {:.4f}'
    TB_LOGGER = SummaryWriter(f"tb_logs/{exp_prefix}")
    print(f"Running experiment: {exp_prefix}")
    
    for epoch in range(0, start_epoch):
        scheduler.step()
    for epoch in range(start_epoch, n_epochs):
        print("Current LR: ",scheduler.get_lr()[0])
        ## Train stage #######################################################################################
        model.train()
        train_loss, metrics = train_epoch(train_loader, model, loss_fn, optimizer, cuda, log_interval, metrics,TB_LOGGER)
        message = TRAIN_LOG.format(epoch + 1, n_epochs, train_loss)
        TB_LOGGER.add_scalar('Loss/Epoch/train', train_loss, epoch + 1)
        for metric in metrics:
            avg_triplets=metric.value()
            message += METRIC_LOG.format(metric.name(), avg_triplets)
        TB_LOGGER.add_scalar('Avg_Non_Zero_Triplets/Epoch/train', avg_triplets, epoch + 1)
        
        ## Val stage  ##########################################################
        model.eval()
        val_loss, metrics = test_epoch(val_loader, model, loss_fn, cuda, metrics,TB_LOGGER)
        TB_LOGGER.add_scalar('Loss/Epoch/val', val_loss, epoch + 1)
        message += VAL_LOG.format(epoch + 1, n_epochs,val_loss)
        for metric in metrics:
            avg_triplets=metric.value()
            message += METRIC_LOG.format(metric.name(), avg_triplets)
            TB_LOGGER.add_scalar('Avg_Non_Zero_Triplets/Epoch/val', avg_triplets, epoch + 1)
        print(message)
        TB_LOGGER.add_scalar('Learning_rate', scheduler.get_lr()[0], epoch + 1)
        scheduler.step()




def train_epoch(train_loader, model, loss_fn, optimizer, cuda, log_interval, metrics,TB_LOGGER):
    MINI_BATCH_LOG='Train: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'
    METRIC_LOG='\t{}: {}'
    for metric in metrics:
        metric.reset()
    model.train()
    losses = []
    total_loss = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        target = target if len(target) > 0 else None
        if not type(data) in (tuple, list):
            data = (data,)
        if cuda:
            data = tuple(d.cuda() for d in data)
            if target is not None:
                target = target.cuda()


        optimizer.zero_grad()
        outputs = model(*data)

        if type(outputs) not in (tuple, list):
            outputs = (outputs,)

        loss_inputs = outputs
        if target is not None:
            target = (target,)
            loss_inputs += target

        loss_outputs = loss_fn(*loss_inputs)
        loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
        losses.append(loss.item())
        #TB_LOGGER.add_scalar('Loss/train', loss.item())
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

        for metric in metrics:
            metric(outputs, target, loss_outputs)

        if batch_idx % log_interval == 0:
            message = MINI_BATCH_LOG.format(batch_idx * len(data[0]),
                                            len(train_loader.dataset),
                                            100. * batch_idx / len(train_loader),
                                            np.mean(losses))
            for metric in metrics:
                message += METRIC_LOG.format(metric.name(),metric.value())
            print(message)
            losses = []

    total_loss /= (batch_idx + 1)
    return total_loss, metrics




def test_epoch(val_loader, model, loss_fn, cuda, metrics,TB_LOGGER):
    with torch.no_grad():
        for metric in metrics:
            metric.reset()
        model.eval()
        val_loss = 0
        for batch_idx, (data, target) in enumerate(val_loader):
            target = target if len(target) > 0 else None
            if not type(data) in (tuple, list):
                data = (data,)
            if cuda:
                data = tuple(d.cuda() for d in data)
                if target is not None:
                    target = target.cuda()

            outputs = model(*data)

            if type(outputs) not in (tuple, list):
                outputs = (outputs,)
            loss_inputs = outputs
            if target is not None:
                target = (target,)
                loss_inputs += target
            loss_outputs = loss_fn(*loss_inputs)
            loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
            val_loss += loss.item()
            for metric in metrics:
                metric(outputs, target, loss_outputs)
    val_loss /= (batch_idx + 1)
    return val_loss, metrics




import torch.nn.functional as F
from collections import OrderedDict
def conv_res(in_channels, out_channels,*args, **kwargs):
    return nn.Sequential(nn.BatchNorm2d(in_channels),nn.Conv2d(in_channels, out_channels, *args, **kwargs))

def conv_block(in_channels, out_channels):
    return nn.Sequential(OrderedDict([('bnorm',nn.BatchNorm2d(in_channels)),
                                              ('conv', nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=1,padding=1)),
                                              ('conv2',nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=1,padding=1)),
                                              ('conv3',nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=1,padding=1)),
                                              ('temp',nn.Conv2d(out_channels,out_channels,kernel_size=1,stride=1))
                                     ]))




class EmbeddingNet(nn.Module):
    ### TODO: choose an architecture, and complete the class
    def __init__(self,in_channels,out_channels):
        super(EmbeddingNet, self).__init__()
        ## Define layers of a CNN
        self.in_channels=in_channels
        self.out_channels=out_channels
        self.conv_block1=conv_block(in_channels,16)
        self.conv_resIn=conv_res(in_channels,self.conv_block1.conv3.out_channels,kernel_size=1,stride=1)
       
        self.conv_block2=conv_block(self.conv_block1.conv3.out_channels*2,
                                    self.conv_block1.conv3.out_channels*2)
        self.conv_resB1=conv_res(self.conv_block1.conv3.out_channels,
                                 self.conv_block2.conv3.out_channels,kernel_size=1,stride=1)
        
        self.conv_block3=conv_block(self.conv_block2.conv3.out_channels*2,
                                    self.conv_block2.conv3.out_channels*2)
        self.conv_resB2=conv_res(self.conv_block2.conv3.out_channels,
                                 self.conv_block3.conv3.out_channels, kernel_size=1,stride=1)

        self.conv_block4=conv_block(self.conv_block3.conv3.out_channels*2,
                                    self.conv_block3.conv3.out_channels*2)
        self.conv_resB3=conv_res(self.conv_block3.conv3.out_channels,
                                 self.conv_block4.conv3.out_channels, kernel_size=1,stride=1)
        
        self.conv_block5=conv_block(self.conv_block4.conv3.out_channels*2,self.conv_block4.conv3.out_channels*2)

        self.MaxPool=nn.MaxPool2d(2, 2)
        self.AdpMaxPool=nn.AdaptiveMaxPool2d((4,4))
        
        self.classifier=nn.Sequential(nn.BatchNorm2d(self.conv_block5.conv3.out_channels),
                                      nn.Conv2d(self.conv_block5.conv3.out_channels, 1024, 4),
                                      nn.BatchNorm2d(1024),
                                      nn.SELU(),
                                      nn.AlphaDropout(0.40),
                                      nn.Conv2d(1024, self.out_channels, 1)
                             )
    def forward(self, x):
        h1=self.MaxPool(F.relu((self.conv_block1(x))))
        res_I=self.MaxPool(self.conv_resIn(x))
        
        h2=self.MaxPool(F.relu((self.conv_block2(torch.cat([h1,res_I],dim=1)))))#concatenate in dimension
        res_h1=self.MaxPool(self.conv_resB1(h1))
        
        h3=self.MaxPool(F.relu((self.conv_block3(torch.cat([h2,res_h1],dim=1)))))#concatenate in dimension
        res_h2=self.MaxPool(self.conv_resB2(h2))
        
        h4=self.MaxPool(F.relu((self.conv_block4(torch.cat([h3,res_h2],dim=1)))))#concatenate in dimension
        res_h3=self.MaxPool(self.conv_resB3(h3))
        
        h5=self.AdpMaxPool(F.relu((self.conv_block5(torch.cat([h4,res_h3],dim=1)))))#concatenate in dimension
        return self.classifier(h5)
    def get_embedding(self, x):
        return self.forward(x).view(-1,self.out_channels)




train_dataset = SpectrogramDataset(audio_conf,train, labels=["voicemail","speech"],
                                       normalize=True)
val_dataset = SpectrogramDataset(audio_conf,val, labels=["voicemail","speech"],
                                       normalize=True)




train_batch_sampler = BalancedBatchSampler(train_dataset.targets, n_classes=2, n_samples=4)
val_batch_sampler = BalancedBatchSampler(val_dataset.targets, n_classes=2, n_samples=4)




train_loader = AudioDataLoader(train_dataset,batch_sampler=train_batch_sampler)
val_loader = AudioDataLoader(val_dataset,batch_sampler=val_batch_sampler)




train_dataset = SpectrogramDataset(audio_conf,train, labels=["voicemail","speech"],
                                       normalize=True)
val_dataset = SpectrogramDataset(audio_conf,val, labels=["voicemail","speech"],
                                       normalize=True)
train_loader = AudioDataLoader(train_dataset,batch_sampler=BalancedBatchSampler(train_dataset.targets, n_classes=2, n_samples=4))
val_loader = AudioDataLoader(val_dataset,batch_sampler= BalancedBatchSampler(val_dataset.targets, n_classes=2, n_samples=4))




class OnlineTripletLoss_lrFinder(OnlineTripletLoss):

    def __init__(self, margin, triplet_selector):
        super(OnlineTripletLoss_lrFinder, self).__init__(margin,triplet_selector)

    def forward(self, embeddings, target):
        return super().forward(embeddings,target)[0]




cuda = torch.cuda.is_available()
model = EmbeddingNet(1,512)
if cuda:
    model.cuda()




from torchsummary import summary
summary(model,torch.rand(1, 64, 735).shape)




from torch.optim import Optimizer
"""TAKEN FROM NVIDIA DL SAMPLES, https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/SpeechRecognition/Jasper/optimizers.py"""
class Novograd(Optimizer):
    """
    Implements Novograd algorithm.
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.95, 0))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        grad_averaging: gradient averaging
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False)
    """

    def __init__(self, params, lr=1e-3, betas=(0.95, 0), eps=1e-8,
                 weight_decay=0, grad_averaging=False, amsgrad=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                      weight_decay=weight_decay,
                      grad_averaging=grad_averaging,
                      amsgrad=amsgrad)

        super(Novograd, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(Novograd, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
            and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Sparse gradients are not supported.')
                amsgrad = group['amsgrad']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros([]).to(state['exp_avg'].device)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros([]).to(state['exp_avg'].device)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                norm = torch.sum(torch.pow(grad, 2))

                if exp_avg_sq == 0:
                    exp_avg_sq.copy_(norm)
                else:
                    exp_avg_sq.mul_(beta2).add_(1 - beta2, norm)

                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                else:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])

                grad.div_(denom)
                if group['weight_decay'] != 0:
                    grad.add_(group['weight_decay'], p.data)
                if group['grad_averaging']:
                    grad.mul_(1 - beta1)
                exp_avg.mul_(beta1).add_(grad)

                p.data.add_(-group['lr'], exp_avg)
        
        return loss




from torch_lr_finder import LRFinder
import torch.optim as optim
margin = 1.
loss_fn = OnlineTripletLoss_lrFinder(margin, RandomNegativeTripletSelector(margin,cpu=False))
optimizer =  Novograd(model.parameters(), lr=1e-6,weight_decay=1e-5)
lr_finder = LRFinder(model, optimizer, loss_fn, device="cuda")
lr_finder.range_test(train_loader, val_loader=val_loader, end_lr=1, num_iter=100, step_mode="exp")




get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")
lr_finder.plot(log_lr=True)




cuda = torch.cuda.is_available()
model = EmbeddingNet(1,512)
if cuda:
    model.cuda()




import torch.optim as optim
from torch.optim import lr_scheduler
margin = 1.
loss_fn = OnlineTripletLoss(margin, RandomNegativeTripletSelector(margin,cpu=False))
lr = 1e-3
optimizer = Novograd(model.parameters(), lr=lr,weight_decay=1e-5)
scheduler = lr_scheduler.StepLR(optimizer,8, gamma=0.95, last_epoch=-1)
n_epochs = 25
log_interval = 50
EXPERIMENT_ID='densenet_novograd512'




LOG_DIR = 'tb_logs'
get_ipython().system_raw(
    'tensorboard --logdir {} --host 0.0.0.0 --port 6006 &'
    .format(LOG_DIR)
)
get_ipython().system_raw('./ngrok http 6006 &')
get_ipython().system(' curl -s http://localhost:4040/api/tunnels | python3 -c     "import sys, json; print(json.load(sys.stdin)[\'tunnels\'][0][\'public_url\'])"')




fit(train_loader,
    val_loader,
    model,
    loss_fn,
    optimizer,
    scheduler,
    n_epochs,
    cuda,
    log_interval,
    exp_prefix=EXPERIMENT_ID,
    metrics=[AverageNonzeroTripletsMetric()])




classes = ["voicemail","speech","silence","beep"]
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
              '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
              '#bcbd22', '#17becf']
def plot_embeddings(embeddings, targets,pca=None, xlim=None, ylim=None):
    if pca:
        embeddings = pca.transform(embeddings)
    plt.figure(figsize=(10,10))
    for i in range(10):
        inds = np.where(targets==i)[0]
        plt.scatter(embeddings[inds,0], embeddings[inds,1], alpha=0.2, color=colors[i])
    if xlim:
        plt.xlim(xlim[0], xlim[1])
    if ylim:
        plt.ylim(ylim[0], ylim[1])
    plt.legend(classes)
    
def extract_embeddings(dataloader, model):
    with torch.no_grad():
        model.eval()
        embeddings = []
        labels =  []
        k = 0
        for images, target in dataloader:
            if cuda:
                images = images.cuda()
            embeddings.extend(model.get_embedding(images).data.cpu().numpy())
            labels.extend(target.numpy())
            k += len(images)
    return np.array(embeddings), np.array(labels)




train_embeddings, train_labels = extract_embeddings(train_loader, model)
val_embeddings, val_labels = extract_embeddings(val_loader, model)




from sklearn.decomposition import PCA
pca = PCA(n_components=2, whiten=True)
pca.fit(np.vstack([train_embeddings,val_embeddings]))




plot_embeddings(train_embeddings, train_labels,pca=pca)
plot_embeddings(val_embeddings, val_labels,pca=pca)




torch.save(model.state_dict(), 'densenet_512_novograd_final.pt')




get_ipython().system('tar -zcvf densenet_512_novograd_final.tar.gz tb_logs/')




class Voicemail_net(nn.Module):
    ### TODO: choose an architecture, and complete the class
    def __init__(self,EmbeddingNet,output_dim=2):
        super(Voicemail_net, self).__init__()
        ## Define layers of a CNN
        self.EmbeddingNet=EmbeddingNet
        self.input_channels=EmbeddingNet.out_channels
        self.output_dim=output_dim
        self.classifier=nn.Sequential(nn.BatchNorm2d(self.input_channels),
                                      nn.Conv2d(self.input_channels,256,1),
                                      nn.BatchNorm2d(256),
                                      nn.SELU(),
                                      nn.AlphaDropout(0.40),
                                      nn.Conv2d(256, self.output_dim, 1),
                             )
        self.activation=nn.LogSoftmax(dim=1)
    def forward(self, x):
        embeddings=self.EmbeddingNet(x)
        logits=self.classifier(embeddings)
        return self.activation(logits.view(-1,self.output_dim))




train_dataset = SpectrogramDataset(audio_conf,train, labels=["voicemail","speech"],
                                       normalize=True)
val_dataset = SpectrogramDataset(audio_conf,val, labels=["voicemail","speech"],
                                       normalize=True)
train_batch_sampler = BalancedBatchSampler(train_dataset.targets, n_classes=2, n_samples=4)
val_batch_sampler = BalancedBatchSampler(val_dataset.targets, n_classes=2, n_samples=4)
train_loader = AudioDataLoader(train_dataset,batch_sampler=train_batch_sampler)
val_loader = AudioDataLoader(val_dataset,batch_sampler=val_batch_sampler)

dataloaders={'train':train_loader,'val':val_loader}




import copy
em_net=copy.deepcopy(model)




cuda = torch.cuda.is_available()
class_model = Voicemail_net(em_net)
if cuda:
    class_model.cuda()




for param in class_model.EmbeddingNet.parameters():
    param.requires_grad = False
class_model.EmbeddingNet=class_model.EmbeddingNet.eval()




from torch_lr_finder import LRFinder
import torch.optim as optim
margin = 1.
loss_fn = nn.NLLLoss()
optimizer =  Novograd(class_model.parameters(), lr=1e-6,weight_decay=1e-5)
lr_finder = LRFinder(class_model, optimizer, loss_fn, device="cuda")
lr_finder.range_test(train_loader, val_loader=val_loader, end_lr=1, num_iter=100, step_mode="exp")




lr_finder.plot(log_lr=True)




def train_model(model, criterion, optimizer, scheduler,dataloaders, num_epochs=25,
                device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    dataset_sizes={'val':dataloaders['val'].dataset.size,'train':dataloaders['train'].dataset.size}
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    log_probs = model(inputs)
                    _, preds = torch.max(torch.exp(log_probs), 1)
                    loss = criterion(log_probs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model




em_net=copy.deepcopy(model)
cuda = torch.cuda.is_available()
class_model = Voicemail_net(em_net)

if cuda:
    class_model.cuda()
    
for param in class_model.EmbeddingNet.parameters():
    param.requires_grad = False
class_model.EmbeddingNet=class_model.EmbeddingNet.eval()




criterion = nn.NLLLoss()
optimizer =  Novograd(class_model.parameters(), lr=0.0008,weight_decay=1e-5)#0.0007
scheduler = lr_scheduler.StepLR(optimizer,8, gamma=0.97, last_epoch=-1)




best=train_model(class_model, criterion, optimizer, scheduler,dataloaders, num_epochs=25)




torch.save(class_model.state_dict(), 'complete_classifier_v1.pt')




torch.save(class_model.classifier.state_dict(), 'only_classifier_v1.pt')




test_loader = AudioDataLoader(test_dataset,batch_sampler=test_batch_sampler)




class_model.classifier.eval()
class_model.EmbeddingNet.eval()

labels =  []
gold = []
for audio, target in test_loader:
    if cuda:
        audio = audio.cuda()
    with torch.no_grad():
        _, preds = torch.max(torch.exp(class_model(audio)), 1)
        labels.extend(preds.cpu().numpy())
        gold.extend(target.numpy())




import seaborn as sns
from sklearn.metrics import confusion_matrix
plt.figure(figsize=(10, 10))
sns.heatmap(confusion_matrix(gold, labels), 
                xticklabels=['voicemail','speech'],
                yticklabels=['voicemail','speech'],annot=True)
plt.show()




get_ipython().system('pip install plot-metric')




from plot_metric.functions import BinaryClassification
bc = BinaryClassification(gold, labels, labels=["voicemail", "speech"])
# Figures
plt.figure(figsize=(5,5))
bc.plot_roc_curve()
plt.show()




bc.print_report()




def detect_voice_segments(vad, audio, sample_rate, frame_bytes, frame_duration_ms, triggered_sliding_window_threshold = 0.9):
    padding_duration_ms = frame_duration_ms * 10
    num_padding_frames = int(padding_duration_ms / frame_duration_ms)
    ring_buffer = collections.deque(maxlen=num_padding_frames)
    triggered = False
    makeseg = lambda voiced_frames: b''.join(voiced_frames)
    voiced_frames = []
    for frame in (audio[offset:offset + frame_bytes] for offset in range(0, len(audio), frame_bytes) if offset + frame_bytes < len(audio)):
        is_speech = vad.is_speech(frame, sample_rate)
        if not triggered:
            ring_buffer.append((frame, is_speech))
            num_voiced = len([f for f, speech in ring_buffer if speech])
            if num_voiced > triggered_sliding_window_threshold * ring_buffer.maxlen:
                triggered = True
                for f, s in ring_buffer:
                    voiced_frames.append(f)
                ring_buffer.clear()
        else:
            voiced_frames.append(frame)
            ring_buffer.append((frame, is_speech))
            num_unvoiced = len([f for f, speech in ring_buffer if not speech])
            if num_unvoiced > triggered_sliding_window_threshold * ring_buffer.maxlen:
                triggered = False
                yield makeseg(voiced_frames)
                ring_buffer.clear()
                voiced_frames = []
    if triggered:
        pass
    if voiced_frames:
        yield makeseg(voiced_frames)




get_ipython().system('apt install -y ffmpeg && pip install webrtcvad pydub')




import subprocess
import webrtcvad
import collections




class InferenceSpectrogramParser():    
    def __init__(self, audio_conf,vad_config,vad_generator_fn, normalize=False,mono=True,filterbanks=64):
        """
        Parses audio file into spectrogram with optional normalization and various augmentations
        :param audio_conf: Dictionary containing the sample rate, window and the window length/stride in seconds
        :param normalize(default False):  Apply standard mean and deviation normalization to audio tensor
        :param speed_volume_perturb(default False): Apply random tempo and gain perturbations
        :param spec_augment(default False): Apply simple spectral augmentation to mel spectograms
        """
        super(InferenceSpectrogramParser, self).__init__()
        self.windows = {'hamming': torch.hamming_window,
                        'hann': torch.hann_window,
                        'blackman': torch.blackman_window,
                        'bartlett': torch.bartlett_window}
        self.window_stride = audio_conf['window_stride']
        self.window_size = audio_conf['window_size']
        self.sample_rate = audio_conf['sample_rate']
        self.window = self.windows.get(audio_conf['window'], None)
        self.mono=mono
        self.normalize = normalize
        self.n_fft=int(self.sample_rate * self.window_size)
        self.hop_length=int(self.sample_rate * self.window_stride)
        self.filterbanks=filterbanks
        self.vad_aggresive=vad_config['aggresive']
        self.vad_frame_duration_ms=vad_config['frame_duration_ms']
        self.vad_frame_bytes=int(2 * self.sample_rate  * (self.vad_frame_duration_ms / 1000.0))
        self.vad_min_seg_duration=vad_config['min_seg_duration']
        self.ffmeg_command="""ffmpeg -loglevel fatal         -hide_banner -nostats -nostdin -i {} -ar 16000         -f s16le -acodec pcm_s16le -ac 1 -map_channel 0.0.0 {} -vn -"""
        self.resampler="-resampler soxr"
        self.spectransformer=torchaudio.transforms.MelSpectrogram(n_fft=self.n_fft,
                                                                  hop_length=self.hop_length,
                                                                  win_length=self.n_fft,
                                                                  n_mels=self.filterbanks,
                                                                  sample_rate=self.sample_rate,
                                                                  window_fn=self.window)
        self.amptodb=torchaudio.transforms.AmplitudeToDB(stype='power', top_db=80)
        self.detect_voice_segments=vad_generator_fn
    
    def parse_audio(self, audio_path):
        try:
            y_bytes = subprocess.check_output(self.ffmeg_command.format(audio_path,self.resampler),stderr = subprocess.DEVNULL,shell=True)
        except:
            y_bytes = subprocess.check_output(self.ffmeg_command.format(audio_path,''),stderr = subprocess.DEVNULL,shell=True)
        segments = self.detect_voice_segments(webrtcvad.Vad(self.vad_aggresive),
                                              y_bytes,
                                              self.sample_rate,
                                              self.vad_frame_bytes,
                                              self.vad_frame_duration_ms)
        max_length=-1
        freq_size=-1
        outputs=[]
        for segment in segments:
            if len(segment) / (2 * self.sample_rate) > self.vad_min_seg_duration:
                y=librosa.util.buf_to_float(segment,n_bytes=2)
                y=torch.FloatTensor(y)
                mel = self.spectransformer(y)
                spect = self.amptodb(mel)
                if self.normalize:
                    mean = spect.mean()
                    std = spect.std()
                    spect.add_(-mean)
                    spect.div_(std)
                outputs.append(spect)
                if spect.size(1)>max_length:
                    freq_size,max_length=spect.size()
        if len(outputs) > 0:
            spects = torch.zeros(len(outputs), 1, freq_size, max_length)
            for i,spect in enumerate(outputs):
                seq_length=spect.size(1)
                spects[i][0].narrow(1, 0, seq_length).copy_(spect)
            return spects,spects.size(0),audio_path
        else:
            return None,0,audio_path




class InferenceSpectrogramDataset(Dataset, InferenceSpectrogramParser):
    def __init__(self,paths,audio_conf,vad_config,vad_generator_fn,labels=['voicemail,speech'], normalize=False,mono=True):
        """
        Dataset that loads tensors via a csv containing file paths to audio files and transcripts separated by
        a comma. Each new line is a different sample. Example below:
        /path/to/audio.wav,label
        ...
        :param audio_conf: Dictionary containing the sample rate, window and the window length/stride in seconds
        :param manifest_filepath: Path to manifest csv as describe above
        :param labels: String containing all the possible characters to map to
        :param normalize: Apply standard mean and deviation normalization to audio tensor
        :param speed_volume_perturb(default False): Apply random tempo and gain perturbations
        :param spec_augment(default False): Apply simple spectral augmentation to mel spectograms
        """
        self.paths = paths
        self.size = len(paths)
        self.labels_set=set(labels)
        self.labels_map = dict([(labels[i], i) for i in range(len(labels))])
        super(InferenceSpectrogramDataset, self).__init__(audio_conf, vad_config, vad_generator_fn,normalize=normalize,mono=mono)

    def __getitem__(self, index):
        audio_path = self.paths[index]
        spects,vad_n,audio_path  = self.parse_audio(audio_path)
        return spects,vad_n,audio_path 

    def __len__(self):
        return self.size




audio_conf = dict(sample_rate=16000,
                          window_size=.025,
                          window_stride=.01,
                          window="hamming")
vad_conf = dict(sample_rate=16000,
                          frame_duration_ms=20,
                          aggresive=3,
                          min_seg_duration=1)




paths=[path for path in glob.glob("/kaggle/input/inferencetest/*.wav")]




inference_dataset=InferenceSpectrogramDataset(paths,audio_conf,vad_conf,detect_voice_segments,normalize=True)




import pandas as pd
def inference(infe_dataset,model,output=None,cuda=torch.cuda.is_available()):
    outputs = pd.DataFrame(columns=['path','prob_speech','vad_prob','vad_n'])
    model.eval()
    for spects,vad_n,audio_path in inference_dataset:
        if spects is not None:
            if cuda:
                spects=spects.cuda()
            with torch.no_grad():
                _, preds = torch.max(torch.exp(class_model(spects)), 1)
                prob_speech=preds.float().mean()
                outputs=outputs.append({'path' : audio_path , 'prob_speech' : prob_speech.item(),'vad_prob':preds.cpu().numpy(),'vad_n':vad_n},ignore_index=True)
        else:
            outputs=outputs.append({'path' : audio_path , 'prob_speech' : None,'vad_prob':None,'vad_n':vad_n},ignore_index=True)
    return outputs




results=inference(inference_dataset,class_model)




results




results.values




for audio_path,prob_speech,preds,vad_n  in results.values:
    print(f"""
    Audio Path: {audio_path} 
    -probability of being speech: {prob_speech}
    -individual probabilities: {preds}
    -Number of Voice Activities: {vad_n}
    {'-'*100}
    """)




cuda = torch.cuda.is_available()

embedding_net = EmbeddingNet(1,512)
if cuda:
    embedding_net.cuda()

classification_model = Voicemail_net(embedding_net)

classification_model.load_state_dict(torch.load("complete_classifier_v1.pt",
                                     map_location=torch.device('cpu')))

if cuda:
    classification_model.cuda()
    
for param in classification_model.parameters():
    param.requires_grad = False
    
classification_model=classification_model.eval()




get_ipython().run_line_magic('time', 'results=inference(inference_dataset,classification_model)')




results




model_trace=copy.deepcopy(classification_model)
model_trace = torch.jit.trace(model_trace,torch.rand(1,1, 64, 735).cuda())




get_ipython().run_line_magic('time', 'results=inference(inference_dataset,model_trace)')




results






