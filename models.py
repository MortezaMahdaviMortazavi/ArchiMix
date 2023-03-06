<<<<<<< HEAD
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torchvision
from torch.autograd import Variable
from config import *
from gensim.models import KeyedVectors
# from config import *
from config import VOCAB_SIZE , EMBEDDING_DIM , GLOVE_PATH , WORD2IDX
from vocabulary import *


class EmbeddingLayer(nn.Module):
    def __init__(self, vocab_size, embedding_dim, glove_path, word2idx):
        super(EmbeddingLayer, self).__init__()

        # Load pre-trained GloVe embeddings
        glove_vectors = {}
        with open(glove_path, 'r', encoding='utf-8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                vector = torch.tensor([float(val) for val in values[1:]])
                glove_vectors[word] = vector

        # Create embedding layer
        self.embedding_layer = nn.Embedding(vocab_size, embedding_dim)

        # Initialize embedding layer with GloVe vectors
        for word, vector in glove_vectors.items():
            if word in word2idx:
                index = word2idx[word]
                self.embedding_layer.weight.data[index] = vector

    def forward(self, inputs):
        embeddings = self.embedding_layer(inputs)
        # Rest of the model architecture
        return embeddings

    

class Encoder(nn.Module):
    def __init__(self,encoded_img_size):
        super(Encoder,self).__init__()
        self.resnet = torch.load('saved_models/resnet101.pt')
        # modules = list(resnet.children())[:-1]
        self.encoded_img_size = encoded_img_size
        # self.resnet = None
        self.out_features = None

    def make_no_grad(self,module):
        for param in module.parameters():
            param.requires_grad = False
            
    def make_trainable(self,module):
        for param in module.parameters():
            param.requires_grad = True
        
    def resnet_modifier(self):
        self.make_no_grad(self.resnet.conv1)
        self.make_no_grad(self.resnet.bn1)
        self.make_no_grad(self.resnet.relu)
        self.make_no_grad(self.resnet.maxpool)
        self.make_no_grad(self.resnet.layer1)
        self.make_no_grad(self.resnet.layer2)

        self.make_trainable(self.resnet.layer3)
        self.make_trainable(self.resnet.layer4)

        # Resize image to fixed size to allow input images of variable size
        self.resnet.adaptive_pooling = nn.AdaptiveAvgPool2d((self.encoded_img_size,self.encoded_img_size))

        # def forward_imp(img):
            # with torch.no_grad():
            #     out = self.resnet.conv1(img)
            #     out = self.resnet.maxpool(self.resnet.relu(self.resnet.bn1(out)))
            #     out = self.resnet.layer1(out)
            #     out = self.resnet.layer2(out)
            # out = self.resnet.layer3(out)
            # out = self.resnet.layer4(out)
            # out = self.resnet.adaptive_pooling(out)
            # return out

        # self.resnet.forward = forward_imp

    def forward(self,img):
        with torch.no_grad():
            out = self.resnet.conv1(img)
            out = self.resnet.maxpool(self.resnet.relu(self.resnet.bn1(out)))
            out = self.resnet.layer1(out)
            out = self.resnet.layer2(out)
        out = self.resnet.layer3(out)
        out = self.resnet.layer4(out)
        out = self.resnet.adaptive_pooling(out)
        return out


class Attention(torch.nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()

        self.attn = torch.nn.Sequential(
            torch.nn.Linear(enc_hid_dim + 2 * dec_hid_dim, dec_hid_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(dec_hid_dim, 1, bias = False)
        )
        
    def forward(self, hidden, encoder_outputs):
        # hidden            2   B   dec_hid_dim
        # encoder_outputs   L   B   enc_hid_dim
        
        batch_size = encoder_outputs.shape[1]
        src_len = encoder_outputs.shape[0]
        
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        # encoder_outputs   B   L   enc_hid_dim

        #repeat decoder hidden state src_len times
        hidden = torch.cat((hidden[0], hidden[1]), dim=1).unsqueeze(1).repeat(1, src_len, 1)
        # hidden            B   L   dec_hid_dim*2

        return F.softmax(self.attn(torch.cat((encoder_outputs, hidden), dim=2)).squeeze(2), dim=1)



class Decoder(nn.Module):
    def __init__(self,input_dim,hidden_dim,num_layers,attention,dropout=0.33):
        super(Decoder,self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        # self.attention = Attention(input_size=input_dim,
        #                             hidden_size=hidden_dim,
        #                             output_size=hidden_dim   
        #                         )
        self.attention = attention
        self.embedding = EmbeddingLayer(vocab_size=VOCAB_SIZE,
                                        embedding_dim=EMBEDDING_DIM,
                                        glove_path=GLOVE_PATH,
                                        word2idx=WORD2IDX
                                        )
        
        self.lstm = nn.LSTM(input_size=hidden_dim,
                            hidden_size=hidden_dim,
                            num_layers=num_layers,
                            bidirectional=True
                        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,out_features=input_dim)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self,inp,encoder_outputs,hidden,cell):
        embedded = self.embedding(inp)
        print(f"embedding output shape is {embedded.shape}")
        attn_weights = self.attention(hidden,encoder_outputs).unsqueeze(1)
        print(f"attn weights output shape is {attn_weights.shape}")
        encoder_outputs = encoder_outputs.permute(1,0,2)
        print(f"encoder output shape is {encoder_outputs.shape}")
        weighted = torch.bmm(attn_weights,encoder_outputs).permute(1,0,2)
        print(f"weighted output shape is {weighted.shape}")
        lstm_input = torch.cat((embedded,weighted),dim=2)
        print(f"lstm input output shape is {lstm_input.shape}")
        lstm_output , (hidden , cell) = self.lstm(lstm_input,(hidden,cell))
        print(f"lstm output shape is {lstm_output.shape}")
        fc_input = torch.cat((lstm_input,lstm_output),dim=2)
        print(f"fc input output shape is {fc_input.shape}")
        pred = self.fc(fc_input)
        print(f"pred output shape is {pred.shape}")
        return pred , hidden , cell

    def init_hidden(self,batch_size,bidirectional=False):
        # hidden = Variable
        hidden = Variable(torch.zeros(
            2*self.num_layers if bidirectional else self.num_layers,
            batch_size,
            self.hidden_dim
            )
        )
        cell = Variable(torch.zeros(
            2*self.num_layers if bidirectional else self.num_layers,
            batch_size,
            self.hidden_dim
            )
        )
        return hidden , cell
        

class ArchiMix(nn.Module):
    def __init__(self,encoder,decoder):
        super(ArchiMix,self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = DEVICE
        self.vocab = FLICKER30_VOCAB
        
    def forward(self,img,decoder_hidden,decoder_cell,capt_tensor):
        img_features = self.encoder(img)
        capt_tensor = capt_tensor.unsqueeze(1)
        decoder_input = torch.cat((img_features,capt_tensor))
        decoder_output , decoder_hidden , decoder_cell = self.decoder(inp=decoder_input,
                                      encoder_outputs=img_features,
                                      hidden=decoder_hidden,
                                      cell=decoder_cell)
        return decoder_output , decoder_hidden , decoder_cell

        



class RNN(nn.Module):
    def __init__(self,input_dim,hidden_dim,num_layers,dropout=0.33):
        super(RNN,self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers
        self.embedding = EmbeddingLayer(
            vocab_size=VOCAB_SIZE,
            embedding_dim=EMBEDDING_DIM,
            glove_path=GLOVE_PATH,
            word2idx=WORD2IDX
        )
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers
        ) # hidden dim should be equal to Embedding_dim because embedding_dim is output of embedding layer
        self.dropout = nn.Dropout(dropout)

    def forward(self,inp,hidden):
        out = self.embedding(inp)
        out = self.dropout(out)
        out , hidden = self.gru(out)
        out = self.dropout(out)
        return out , hidden

    def init_hidden(self,batch_size):
        hidden = Variable(torch.zeros(self.num_layers,batch_size,self.hidden_dim))
        return hidden
    


if __name__ == "__main__":
    img = torch.randn(1,3,224,224)
    caption = torch.randn(100,1,100)
    print(caption.shape)
    encoder = Encoder(224)
    encoder.resnet_modifier()
    attention = Attention(enc_hid_dim=512,dec_hid_dim=100)
    decoder = Decoder(input_dim=1000,hidden_dim=100,num_layers=3,attention=attention)
    archimix = ArchiMix(encoder=encoder,decoder=decoder)
    hidden , cell = decoder.init_hidden(batch_size=1,bidirectional=2)
    archimix(img,hidden,cell,caption)
    print(archimix)
=======
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torchvision
from torch.autograd import Variable
from config import *
from gensim.models import KeyedVectors
# from config import *
from config import VOCAB_SIZE , EMBEDDING_DIM , GLOVE_PATH , WORD2IDX
from vocabulary import *


class EmbeddingLayer(nn.Module):
    def __init__(self, vocab_size, embedding_dim, glove_path, word2idx):
        super(EmbeddingLayer, self).__init__()

        # Load pre-trained GloVe embeddings
        glove_vectors = {}
        with open(glove_path, 'r', encoding='utf-8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                vector = torch.tensor([float(val) for val in values[1:]])
                glove_vectors[word] = vector

        # Create embedding layer
        self.embedding_layer = nn.Embedding(vocab_size, embedding_dim)

        # Initialize embedding layer with GloVe vectors
        for word, vector in glove_vectors.items():
            if word in word2idx:
                index = word2idx[word]
                self.embedding_layer.weight.data[index] = vector

    def forward(self, inputs):
        embeddings = self.embedding_layer(inputs)
        # Rest of the model architecture
        return embeddings

    

class Encoder(nn.Module):
    def __init__(self,encoded_img_size):
        super(Encoder,self).__init__()
        self.resnet = torch.load('saved_models/resnet101.pt')
        # modules = list(resnet.children())[:-1]
        self.encoded_img_size = encoded_img_size
        # self.resnet = None
        self.out_features = None

    def make_no_grad(self,module):
        for param in module.parameters():
            param.requires_grad = False
            
    def make_trainable(self,module):
        for param in module.parameters():
            param.requires_grad = True
        
    def resnet_modifier(self):
        self.make_no_grad(self.resnet.conv1)
        self.make_no_grad(self.resnet.bn1)
        self.make_no_grad(self.resnet.relu)
        self.make_no_grad(self.resnet.maxpool)
        self.make_no_grad(self.resnet.layer1)
        self.make_no_grad(self.resnet.layer2)

        self.make_trainable(self.resnet.layer3)
        self.make_trainable(self.resnet.layer4)

        # Resize image to fixed size to allow input images of variable size
        self.resnet.adaptive_pooling = nn.AdaptiveAvgPool2d((self.encoded_img_size,self.encoded_img_size))

        # def forward_imp(img):
            # with torch.no_grad():
            #     out = self.resnet.conv1(img)
            #     out = self.resnet.maxpool(self.resnet.relu(self.resnet.bn1(out)))
            #     out = self.resnet.layer1(out)
            #     out = self.resnet.layer2(out)
            # out = self.resnet.layer3(out)
            # out = self.resnet.layer4(out)
            # out = self.resnet.adaptive_pooling(out)
            # return out

        # self.resnet.forward = forward_imp

    def forward(self,img):
        with torch.no_grad():
            out = self.resnet.conv1(img)
            out = self.resnet.maxpool(self.resnet.relu(self.resnet.bn1(out)))
            out = self.resnet.layer1(out)
            out = self.resnet.layer2(out)
        out = self.resnet.layer3(out)
        out = self.resnet.layer4(out)
        out = self.resnet.adaptive_pooling(out)
        return out


class Attention(torch.nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()

        self.attn = torch.nn.Sequential(
            torch.nn.Linear(enc_hid_dim + 2 * dec_hid_dim, dec_hid_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(dec_hid_dim, 1, bias = False)
        )
        
    def forward(self, hidden, encoder_outputs):
        # hidden            2   B   dec_hid_dim
        # encoder_outputs   L   B   enc_hid_dim
        
        batch_size = encoder_outputs.shape[1]
        src_len = encoder_outputs.shape[0]
        
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        # encoder_outputs   B   L   enc_hid_dim

        #repeat decoder hidden state src_len times
        hidden = torch.cat((hidden[0], hidden[1]), dim=1).unsqueeze(1).repeat(1, src_len, 1)
        # hidden            B   L   dec_hid_dim*2

        return F.softmax(self.attn(torch.cat((encoder_outputs, hidden), dim=2)).squeeze(2), dim=1)



class Decoder(nn.Module):
    def __init__(self,input_dim,hidden_dim,num_layers,attention,dropout=0.33):
        super(Decoder,self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        # self.attention = Attention(input_size=input_dim,
        #                             hidden_size=hidden_dim,
        #                             output_size=hidden_dim   
        #                         )
        self.attention = attention
        self.embedding = EmbeddingLayer(vocab_size=VOCAB_SIZE,
                                        embedding_dim=EMBEDDING_DIM,
                                        glove_path=GLOVE_PATH,
                                        word2idx=WORD2IDX
                                        )
        
        self.lstm = nn.LSTM(input_size=hidden_dim,
                            hidden_size=hidden_dim,
                            num_layers=num_layers,
                            bidirectional=True
                        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,out_features=input_dim)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self,inp,encoder_outputs,hidden,cell):
        embedded = self.embedding(inp)
        print(f"embedding output shape is {embedded.shape}")
        attn_weights = self.attention(hidden,encoder_outputs).unsqueeze(1)
        print(f"attn weights output shape is {attn_weights.shape}")
        encoder_outputs = encoder_outputs.permute(1,0,2)
        print(f"encoder output shape is {encoder_outputs.shape}")
        weighted = torch.bmm(attn_weights,encoder_outputs).permute(1,0,2)
        print(f"weighted output shape is {weighted.shape}")
        lstm_input = torch.cat((embedded,weighted),dim=2)
        print(f"lstm input output shape is {lstm_input.shape}")
        lstm_output , (hidden , cell) = self.lstm(lstm_input,(hidden,cell))
        print(f"lstm output shape is {lstm_output.shape}")
        fc_input = torch.cat((lstm_input,lstm_output),dim=2)
        print(f"fc input output shape is {fc_input.shape}")
        pred = self.fc(fc_input)
        print(f"pred output shape is {pred.shape}")
        return pred , hidden , cell

    def init_hidden(self,batch_size,bidirectional=False):
        # hidden = Variable
        hidden = Variable(torch.zeros(
            2*self.num_layers if bidirectional else self.num_layers,
            batch_size,
            self.hidden_dim
            )
        )
        cell = Variable(torch.zeros(
            2*self.num_layers if bidirectional else self.num_layers,
            batch_size,
            self.hidden_dim
            )
        )
        return hidden , cell
        

class ArchiMix(nn.Module):
    def __init__(self,encoder,decoder):
        super(ArchiMix,self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = DEVICE
        self.vocab = FLICKER30_VOCAB
        
    def forward(self,img,decoder_hidden,decoder_cell,capt_tensor):
        img_features = self.encoder(img)
        capt_tensor = capt_tensor.unsqueeze(1)
        decoder_input = torch.cat((img_features,capt_tensor))
        decoder_output , decoder_hidden , decoder_cell = self.decoder(inp=decoder_input,
                                      encoder_outputs=img_features,
                                      hidden=decoder_hidden,
                                      cell=decoder_cell)
        return decoder_output , decoder_hidden , decoder_cell

        



class RNN(nn.Module):
    def __init__(self,input_dim,hidden_dim,num_layers,dropout=0.33):
        super(RNN,self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers
        self.embedding = EmbeddingLayer(
            vocab_size=VOCAB_SIZE,
            embedding_dim=EMBEDDING_DIM,
            glove_path=GLOVE_PATH,
            word2idx=WORD2IDX
        )
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers
        ) # hidden dim should be equal to Embedding_dim because embedding_dim is output of embedding layer
        self.dropout = nn.Dropout(dropout)

    def forward(self,inp,hidden):
        out = self.embedding(inp)
        out = self.dropout(out)
        out , hidden = self.gru(out)
        out = self.dropout(out)
        return out , hidden

    def init_hidden(self,batch_size):
        hidden = Variable(torch.zeros(self.num_layers,batch_size,self.hidden_dim))
        return hidden
    


if __name__ == "__main__":
    img = torch.randn(1,3,224,224)
    caption = torch.randn(100,1,100)
    print(caption.shape)
    encoder = Encoder(224)
    encoder.resnet_modifier()
    attention = Attention(enc_hid_dim=512,dec_hid_dim=100)
    decoder = Decoder(input_dim=1000,hidden_dim=100,num_layers=3,attention=attention)
    archimix = ArchiMix(encoder=encoder,decoder=decoder)
    hidden , cell = decoder.init_hidden(batch_size=1,bidirectional=2)
    archimix(img,hidden,cell,caption)
    print(archimix)
>>>>>>> f72c2826b6880f5d1c126923f46f030081e3e866
