<<<<<<< HEAD
import torch
import torchvision.transforms as transforms
# import torchtext.transforms as transforms
from vocabulary import *
from config import FLICKER30_VOCAB,FLICKER8K_VOCAB,WORD2IDX,IDX2WORD
import json
# def preprocess_images(tranforms):
    

class CaptionTransformer:
    def __init__(self, sequence_length, vocabulary):
        self.sequence_length = sequence_length
        self.word2idx = vocabulary.get_word2idx

    def __call__(self, captions):
        if isinstance(captions, list):
            # Concatenate the tensors for each caption
            tensors = [self.__call__(caption) for caption in captions]
            return torch.stack(tensors)

        # Handle single caption
        tokens = captions.split()
        tensor = torch.zeros(self.sequence_length).long()
        for idx in range(self.sequence_length):
            if idx < len(tokens):
                word = tokens[idx]
                if word in self.word2idx:
                    tensor[idx] = self.word2idx[word]
                else:
                    tensor[idx] = self.word2idx['<unk>']
            else:
                tensor[idx] = self.word2idx['<pad>']
        return tensor

    def __repr__(self):
        return f"CaptionTransformer(sequence_length={self.sequence_length})"

        

class ImageTransformer:
    def __init__(self, img_size=224, color_jitter=True, rotation_degrees=15, translation=(0.1, 0.1), blur=True):
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomResizedCrop(img_size),
            transforms.RandomHorizontalFlip(),
        ])
        if color_jitter:
            self.transform.transforms.append(transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1))
        if rotation_degrees > 0:
            self.transform.transforms.append(transforms.RandomRotation(degrees=rotation_degrees))
        if translation[0] > 0 or translation[1] > 0:
            self.transform.transforms.append(transforms.RandomAffine(degrees=0, translate=translation))
        if blur:
            self.transform.transforms.append(transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)))
        self.transform.transforms.extend([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __call__(self,img):
        out = self.transform(img)
        return out




def save_vocab(flicker30_vocab, flicker8_vocab, path):
    flicker30_filename = path + '/flicker30k_vocab.json'
    flicker8_filename = path + '/flicker8k_vocab.json'
    
    with open(flicker30_filename, 'w', encoding='utf-8') as f:
        json.dump(flicker30_vocab, f, ensure_ascii=False)

    with open(flicker8_filename, 'w', encoding='utf-8') as f:
        json.dump(flicker8_vocab, f, ensure_ascii=False)
        
    print(f'Saved Flicker30K vocab to {flicker30_filename}')
    print(f'Saved Flicker8K vocab to {flicker8_filename}')



# save_vocab(flicker30_vocab=WORD2IDX,flicker8_vocab=IDX2WORD,path='Annotation')



# def calculate_loss():
#     pass


# def calculate_accuracy():
#     pass



=======
import torch
import torchvision.transforms as transforms
# import torchtext.transforms as transforms
from vocabulary import *
from config import FLICKER30_VOCAB,FLICKER8K_VOCAB,WORD2IDX,IDX2WORD
import json
# def preprocess_images(tranforms):
    

class CaptionTransformer:
    def __init__(self, sequence_length, vocabulary):
        self.sequence_length = sequence_length
        self.word2idx = vocabulary.get_word2idx

    def __call__(self, captions):
        if isinstance(captions, list):
            # Concatenate the tensors for each caption
            tensors = [self.__call__(caption) for caption in captions]
            return torch.stack(tensors)

        # Handle single caption
        tokens = captions.split()
        tensor = torch.zeros(self.sequence_length).long()
        for idx in range(self.sequence_length):
            if idx < len(tokens):
                word = tokens[idx]
                if word in self.word2idx:
                    tensor[idx] = self.word2idx[word]
                else:
                    tensor[idx] = self.word2idx['<unk>']
            else:
                tensor[idx] = self.word2idx['<pad>']
        return tensor

    def __repr__(self):
        return f"CaptionTransformer(sequence_length={self.sequence_length})"

        

class ImageTransformer:
    def __init__(self, img_size=224, color_jitter=True, rotation_degrees=15, translation=(0.1, 0.1), blur=True):
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomResizedCrop(img_size),
            transforms.RandomHorizontalFlip(),
        ])
        if color_jitter:
            self.transform.transforms.append(transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1))
        if rotation_degrees > 0:
            self.transform.transforms.append(transforms.RandomRotation(degrees=rotation_degrees))
        if translation[0] > 0 or translation[1] > 0:
            self.transform.transforms.append(transforms.RandomAffine(degrees=0, translate=translation))
        if blur:
            self.transform.transforms.append(transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)))
        self.transform.transforms.extend([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __call__(self,img):
        out = self.transform(img)
        return out




def save_vocab(flicker30_vocab, flicker8_vocab, path):
    flicker30_filename = path + '/flicker30k_vocab.json'
    flicker8_filename = path + '/flicker8k_vocab.json'
    
    with open(flicker30_filename, 'w', encoding='utf-8') as f:
        json.dump(flicker30_vocab, f, ensure_ascii=False)

    with open(flicker8_filename, 'w', encoding='utf-8') as f:
        json.dump(flicker8_vocab, f, ensure_ascii=False)
        
    print(f'Saved Flicker30K vocab to {flicker30_filename}')
    print(f'Saved Flicker8K vocab to {flicker8_filename}')



# save_vocab(flicker30_vocab=WORD2IDX,flicker8_vocab=IDX2WORD,path='Annotation')



# def calculate_loss():
#     pass


# def calculate_accuracy():
#     pass



>>>>>>> f72c2826b6880f5d1c126923f46f030081e3e866
