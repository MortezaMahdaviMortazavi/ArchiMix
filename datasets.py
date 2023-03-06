<<<<<<< HEAD
import torch
import torchvision
from torch.utils.data import Dataset
from vocabulary import *
import os
from torchvision.io import read_image
from PIL import Image
import pandas as pd
from utils import *
# from torch import tensor

"""
Use constants for paths and other variables that do not change during the execution of the program.

Instead of using separate functions to set the images and captions, 
it would be better to define a single function that reads the captions file and stores the images and captions together.

To prevent memory errors, consider using a generator function that returns a batch of images and captions 4
at a time instead of storing all images and captions in memory.

Add a function to save and load the vocabulary to a file 
so that you can reuse it in future runs of your program without having to rebuild the vocabulary from scratch.

Add a function to convert captions to tokenized sequences of integers so that you can use them as inputs to your model.

Add a function to convert tokenized sequences of integers back to captions 
so that you can evaluate the quality of your model's output.

"""

def setCaptions(path):
    df = pd.read_csv(path,delimiter='|')
    captions_dict = {}
    for _ , row in df.iterrows():
        filename = row['image_name']
        caption = row[' comment']
        if filename not in captions_dict:
            captions_dict[filename] = []
        captions_dict[filename].append(caption)
    return captions_dict

s = setCaptions(path='Flickrs/FlickrFullDataset/results.csv')

def getVariable():
    sequence_length = 50
    path = None
    vocabulary = Flickr30kVocabulary(path=path)
    root_dir = None
    captions_dict = setCaptions(path)
    transform = ImageTransformer()
    target_transform = CaptionTransformer(sequence_length=sequence_length)
    return captions_dict,root_dir,transform,target_transform


root_dir = 'Flickrs/FlickrFullDataset/flickr30k_images/'
transform = ImageTransformer()


class Flickr30k(Dataset):
    def __init__(self, captions_dict, root_dir, transform=None, target_transform=None):
        self.captions_dict = captions_dict
        self.filenames = list(captions_dict.keys())
        self.root_dir = root_dir
        self.transform = transform
        self.target_transform = target_transform
        
        # calculate the maximum number of captions and the maximum caption length
        self.max_captions = max([len(captions) for captions in captions_dict.values()])
        self.max_caption_len = max([len(caption) for captions in captions_dict.values() for caption in captions])
        self.target_shape = (self.max_captions, self.max_caption_len)
    
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        filename = self.filenames[idx]
        captions = self.captions_dict[filename]
        image_path = os.path.join(self.root_dir, filename)
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        # transform the list of captions as a whole and then concatenate to form target tensor
        if self.target_transform:
            captions = self.target_transform(captions)
            num_captions = captions.shape[0]
            target = torch.zeros((self.max_captions, self.max_caption_len))
            target[:num_captions, :captions.shape[1]] = captions
        else:
            target = captions
            
        return image, target

            

class Flicker8k(Dataset):
    def __init__(self,annotations,img_dir,transform=None,target_transform=None):
        self.annotations = annotations
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.images = []
        self.targets = []
        with open(annotations,'r') as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split(",")
            self.images.append(parts[0])
            self.targets.append(parts[1])
        self.images.pop(0)
        self.targets.pop(0)

    def freeMemory(self):
        self.images.clear()
        self.targets.clear()
        del self.images
        del self.targets

    def __repr__(self):
        return f'{self.__class__.__name__}(annotations={self.annotations}, img_dir={self.img_dir}, transform={self.transform}, target_transform={self.target_transform}, num_samples={len(self)})'

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir,self.images[index])
        # img_path += '.jpg'
        image = Image.open(img_path)
        target = self.targets[index]

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            target = self.target_transform(target)
        
        return image , target

        

    

class PersianIC(Dataset):
    def __init__(self):
        pass

    def __len__(self):
        pass

    def __getitem__(self, index):
        return super().__getitem__(index)
=======
import torch
import torchvision
from torch.utils.data import Dataset
from vocabulary import *
import os
from torchvision.io import read_image
from PIL import Image
import pandas as pd
from utils import *
# from torch import tensor

"""
Use constants for paths and other variables that do not change during the execution of the program.

Instead of using separate functions to set the images and captions, 
it would be better to define a single function that reads the captions file and stores the images and captions together.

To prevent memory errors, consider using a generator function that returns a batch of images and captions 4
at a time instead of storing all images and captions in memory.

Add a function to save and load the vocabulary to a file 
so that you can reuse it in future runs of your program without having to rebuild the vocabulary from scratch.

Add a function to convert captions to tokenized sequences of integers so that you can use them as inputs to your model.

Add a function to convert tokenized sequences of integers back to captions 
so that you can evaluate the quality of your model's output.

"""

def setCaptions(path):
    df = pd.read_csv(path,delimiter='|')
    captions_dict = {}
    for _ , row in df.iterrows():
        filename = row['image_name']
        caption = row[' comment']
        if filename not in captions_dict:
            captions_dict[filename] = []
        captions_dict[filename].append(caption)
    return captions_dict

s = setCaptions(path='Flickrs/FlickrFullDataset/results.csv')

def getVariable():
    sequence_length = 50
    path = None
    vocabulary = Flickr30kVocabulary(path=path)
    root_dir = None
    captions_dict = setCaptions(path)
    transform = ImageTransformer()
    target_transform = CaptionTransformer(sequence_length=sequence_length)
    return captions_dict,root_dir,transform,target_transform


root_dir = 'Flickrs/FlickrFullDataset/flickr30k_images/'
transform = ImageTransformer()


class Flickr30k(Dataset):
    def __init__(self, captions_dict, root_dir, transform=None, target_transform=None):
        self.captions_dict = captions_dict
        self.filenames = list(captions_dict.keys())
        self.root_dir = root_dir
        self.transform = transform
        self.target_transform = target_transform
        
        # calculate the maximum number of captions and the maximum caption length
        self.max_captions = max([len(captions) for captions in captions_dict.values()])
        self.max_caption_len = max([len(caption) for captions in captions_dict.values() for caption in captions])
        self.target_shape = (self.max_captions, self.max_caption_len)
    
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        filename = self.filenames[idx]
        captions = self.captions_dict[filename]
        image_path = os.path.join(self.root_dir, filename)
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        # transform the list of captions as a whole and then concatenate to form target tensor
        if self.target_transform:
            captions = self.target_transform(captions)
            num_captions = captions.shape[0]
            target = torch.zeros((self.max_captions, self.max_caption_len))
            target[:num_captions, :captions.shape[1]] = captions
        else:
            target = captions
            
        return image, target

            

class Flicker8k(Dataset):
    def __init__(self,annotations,img_dir,transform=None,target_transform=None):
        self.annotations = annotations
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.images = []
        self.targets = []
        with open(annotations,'r') as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split(",")
            self.images.append(parts[0])
            self.targets.append(parts[1])
        self.images.pop(0)
        self.targets.pop(0)

    def freeMemory(self):
        self.images.clear()
        self.targets.clear()
        del self.images
        del self.targets

    def __repr__(self):
        return f'{self.__class__.__name__}(annotations={self.annotations}, img_dir={self.img_dir}, transform={self.transform}, target_transform={self.target_transform}, num_samples={len(self)})'

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir,self.images[index])
        # img_path += '.jpg'
        image = Image.open(img_path)
        target = self.targets[index]

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            target = self.target_transform(target)
        
        return image , target

        

    

class PersianIC(Dataset):
    def __init__(self):
        pass

    def __len__(self):
        pass

    def __getitem__(self, index):
        return super().__getitem__(index)
>>>>>>> f72c2826b6880f5d1c126923f46f030081e3e866
