<<<<<<< HEAD
import os
import pandas as pd

class Flicker8kVocabulary:
    def __init__(self,path):
        self.path = path #'Flickrs/Flickr8K/captions.txt'
        self.word2idx = {}
        self.idx2word = {}
        self.word2count = {}

    def separate_captions(self):
        with open(self.path,'r') as f:
            lines = f.readlines() # read all lines of txt file
        
        captions = []

        for line in lines:
            parts = line.strip().split(',') # separate image names and caption texts and append them into their own data structure
            captions.append(parts[1])

        return captions
    
    @property
    def get_word2idx(self):
        return self.word2idx

    def tokenizer(self):
        captions = self.separate_captions()
        idx = 2
        count = 0
        self.word2idx['<unk>'] = 0
        self.word2idx['<pad>'] = 1
        self.idx2word[0] = '<unk>'
        self.idx2word[1] = '<pad>'
        for caption in captions:
            tokens = caption.split()
            for token in tokens:
                if token not in self.word2idx:
                    self.word2idx[token] = idx # if token wasnt exist in word2idx add it and shift index
                    self.idx2word[idx] = token # do the opposite of word2idx and assign index to a token
                    idx+=1

                if token not in self.word2count: # this block for count each word in vocabulary and captions
                    self.word2count[token] = count
                    count+=1


class Flickr30kVocabulary:
    def __init__(self, path, min_word_freq):
        self.annotations = pd.read_csv(path, delimiter='|')
        self.min_word_freq = min_word_freq
        self.word2idx = {"<unk>": 0, "<pad>": 1}
        self.idx2word = {0: "<unk>", 1: "<pad>"}
        self.word2count = {}
    
    def build_vocab(self):
        # Count the frequency of each word
        for i in range(len(self.annotations)):
            comment = self.annotations[' comment'][i]
            if isinstance(comment, str):
                splits = comment.split()
                for token in splits:
                    if token not in self.word2count:
                        self.word2count[token] = 1
                    else:
                        self.word2count[token] += 1
        
        # Add words to vocabulary if their frequency is above the minimum threshold
        idx = 2
        for word, count in self.word2count.items():
            if count >= self.min_word_freq:
                self.word2idx[word] = idx
                self.idx2word[idx] = word
                idx += 1
    
    @property
    def get_word2idx(self):
        return self.word2idx
    
    @property
    def get_idx2word(self):
        return self.idx2word
    
    def __call__(self):
        self.build_vocab()
        word2idx = self.get_word2idx
        idx2word = self.get_idx2word
        return word2idx, idx2word


class PersianVocabulary:
    def __init__(self):
        pass

    def tokenizer(self):
        pass

=======
import os
import pandas as pd

class Flicker8kVocabulary:
    def __init__(self,path):
        self.path = path #'Flickrs/Flickr8K/captions.txt'
        self.word2idx = {}
        self.idx2word = {}
        self.word2count = {}

    def separate_captions(self):
        with open(self.path,'r') as f:
            lines = f.readlines() # read all lines of txt file
        
        captions = []

        for line in lines:
            parts = line.strip().split(',') # separate image names and caption texts and append them into their own data structure
            captions.append(parts[1])

        return captions
    
    @property
    def get_word2idx(self):
        return self.word2idx

    def tokenizer(self):
        captions = self.separate_captions()
        idx = 2
        count = 0
        self.word2idx['<unk>'] = 0
        self.word2idx['<pad>'] = 1
        self.idx2word[0] = '<unk>'
        self.idx2word[1] = '<pad>'
        for caption in captions:
            tokens = caption.split()
            for token in tokens:
                if token not in self.word2idx:
                    self.word2idx[token] = idx # if token wasnt exist in word2idx add it and shift index
                    self.idx2word[idx] = token # do the opposite of word2idx and assign index to a token
                    idx+=1

                if token not in self.word2count: # this block for count each word in vocabulary and captions
                    self.word2count[token] = count
                    count+=1


class Flickr30kVocabulary:
    def __init__(self, path, min_word_freq):
        self.annotations = pd.read_csv(path, delimiter='|')
        self.min_word_freq = min_word_freq
        self.word2idx = {"<unk>": 0, "<pad>": 1}
        self.idx2word = {0: "<unk>", 1: "<pad>"}
        self.word2count = {}
    
    def build_vocab(self):
        # Count the frequency of each word
        for i in range(len(self.annotations)):
            comment = self.annotations[' comment'][i]
            if isinstance(comment, str):
                splits = comment.split()
                for token in splits:
                    if token not in self.word2count:
                        self.word2count[token] = 1
                    else:
                        self.word2count[token] += 1
        
        # Add words to vocabulary if their frequency is above the minimum threshold
        idx = 2
        for word, count in self.word2count.items():
            if count >= self.min_word_freq:
                self.word2idx[word] = idx
                self.idx2word[idx] = word
                idx += 1
    
    @property
    def get_word2idx(self):
        return self.word2idx
    
    @property
    def get_idx2word(self):
        return self.idx2word
    
    def __call__(self):
        self.build_vocab()
        word2idx = self.get_word2idx
        idx2word = self.get_idx2word
        return word2idx, idx2word


class PersianVocabulary:
    def __init__(self):
        pass

    def tokenizer(self):
        pass

>>>>>>> f72c2826b6880f5d1c126923f46f030081e3e866
