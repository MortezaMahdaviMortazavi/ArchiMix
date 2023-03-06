<<<<<<< HEAD
import torch
import os
from vocabulary import *
# from models import ArchiMix


GLOVE_PATH = 'Annotation/glove.6B.100d.txt'
FLICKER8K_CAPTIONS_PATH = 'Flickrs/Flickr8K/captions.txt'
FLICKER30K_CAPTIONS_PATH = 'Flickrs/FlickrFullDataset/results.csv'
PERSIAN_ANNOTATION_PATH = 'Annotation/news.json'
MIN_WORD_FREQ = 5
FLICKER30_VOCAB = Flickr30kVocabulary(path=FLICKER30K_CAPTIONS_PATH,min_word_freq=MIN_WORD_FREQ)
FLICKER8K_VOCAB = Flicker8kVocabulary(path=FLICKER8K_CAPTIONS_PATH)
WORD2IDX = Flickr30kVocabulary(path=FLICKER30K_CAPTIONS_PATH,min_word_freq=MIN_WORD_FREQ)()[0]
IDX2WORD = Flickr30kVocabulary(path=FLICKER30K_CAPTIONS_PATH,min_word_freq=MIN_WORD_FREQ)()[1]
VOCAB_SIZE = len(WORD2IDX)
EMBEDDING_DIM = 100
MODEL_ARCHITECTURE = None
TRAIN_DATA_DIR = None
VAL_DATA_DIR = None
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 256
LEARNING_RATE = 0.002
WEIGHT_DECAY = 0.0005
NUM_EPOCHS = 2000
NUM_WORKERS = 2
PIN_MEMORY = True
SAVE_MODEL = True
LOAD_MODEL = False
# ARCHIMIX = ArchiMix()
# MODEL_PARAMS = filter(lambda p: p.requires_grad, ARCHIMIX.embedding_layer.parameters()) # Filter out the parameters of the model
# OPTIMIZER = torch.optim.Adam(MODEL_PARAMS, lr=LEARNING_RATE)# Define the optimizer with the filtered parameters and the learning rate
LOSS_FUNCTION = torch.nn.CrossEntropyLoss()
VOCAB_SIZE = len(WORD2IDX)
IMAGE_SIZE = 224
CAPTION_LENGTH = 50
LOG_DIR = None
=======
import torch
import os
from vocabulary import *
# from models import ArchiMix


GLOVE_PATH = 'Annotation/glove.6B.100d.txt'
FLICKER8K_CAPTIONS_PATH = 'Flickrs/Flickr8K/captions.txt'
FLICKER30K_CAPTIONS_PATH = 'Flickrs/FlickrFullDataset/results.csv'
PERSIAN_ANNOTATION_PATH = 'Annotation/news.json'
MIN_WORD_FREQ = 5
FLICKER30_VOCAB = Flickr30kVocabulary(path=FLICKER30K_CAPTIONS_PATH,min_word_freq=MIN_WORD_FREQ)
FLICKER8K_VOCAB = Flicker8kVocabulary(path=FLICKER8K_CAPTIONS_PATH)
WORD2IDX = Flickr30kVocabulary(path=FLICKER30K_CAPTIONS_PATH,min_word_freq=MIN_WORD_FREQ)()[0]
IDX2WORD = Flickr30kVocabulary(path=FLICKER30K_CAPTIONS_PATH,min_word_freq=MIN_WORD_FREQ)()[1]
VOCAB_SIZE = len(WORD2IDX)
EMBEDDING_DIM = 100
MODEL_ARCHITECTURE = None
TRAIN_DATA_DIR = None
VAL_DATA_DIR = None
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 256
LEARNING_RATE = 0.002
WEIGHT_DECAY = 0.0005
NUM_EPOCHS = 2000
NUM_WORKERS = 2
PIN_MEMORY = True
SAVE_MODEL = True
LOAD_MODEL = False
# ARCHIMIX = ArchiMix()
# MODEL_PARAMS = filter(lambda p: p.requires_grad, ARCHIMIX.embedding_layer.parameters()) # Filter out the parameters of the model
# OPTIMIZER = torch.optim.Adam(MODEL_PARAMS, lr=LEARNING_RATE)# Define the optimizer with the filtered parameters and the learning rate
LOSS_FUNCTION = torch.nn.CrossEntropyLoss()
VOCAB_SIZE = len(WORD2IDX)
IMAGE_SIZE = 224
CAPTION_LENGTH = 50
LOG_DIR = None
>>>>>>> f72c2826b6880f5d1c126923f46f030081e3e866
