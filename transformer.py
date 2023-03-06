<<<<<<< HEAD
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable

class ScaleDotProduct(nn.Module):
    def __init__(self,query_dim,key_dim,value_dim):
        super(ScaleDotProduct,self).__init__()
        self.scale = nn.Parameter(torch.Tensor([np.sqrt(key_dim)]))
        
    def forward(self, mask, query, keys, values):
            # query : [B,Q] (hidden_state,decoder_output)
            # Keys : [T,B,K] (encoder outputs)
            # Values : [T,B,V] (encoder outputs)
            # assume Q == K

            # compute energy:
            query = query.unsqueeze(1) # [B,Q] ==> [B,1,Q]
            print(f"Query shape is {query.shape}")
            keys = keys.permute(1,2,0) # [T,B,K] ==> [B,K,T]
            print(f"Keys shape is {keys.shape}")
            energy = torch.bmm(query,keys) # [B,1,Q] * [B,K,T] ==> [B,1,T]
            energy = torch.multiply(self.scale, energy)  # apply scale
            print(f"energy1 shape is {energy.shape}")
            energy = F.softmax(energy,dim=2)
            print(f"energy2 shape is {energy.shape}")

            # apply mask and renormalize
            energy = energy * mask
            energy = energy.div(energy.sum(2,keepdim=True))
            print(f"energy3 shape is {energy.shape}")

            # weight values
            values = values.transpose(0,1) # [T,B,V] -> [B,T,V]
            combo = torch.bmm(energy, values).squeeze(1) # [B,1,T]*[B,T,V] -> [B,V]
            print(f"values shape is {values.shape}")
            print(f"combo shape is {combo.shape}")

            return (combo, energy)



class AdditiveAttention(nn.Module):
    def __init__(self,input_size,hidden_size,output_size,dropout=0.5):
        super(AdditiveAttention,self).__init__()
        self.query = nn.Linear(input_size,hidden_size)
        # self.bias = nn.Parameter()
        self.key = nn.Linear(input_size,hidden_size)
        self.value = nn.Linear(input_size,output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self,inp):
        query = self.query(inp)
        key = self.key(inp)
        attn_weight = torch.matmul(query,key.T)
        attn_weight = F.softmax(attn_weight)
        attn_weight = self.dropout(attn_weight)
        output = self.value(attn_weight)
        return output
    


class PositionalEncoder(nn.Module):
    """
        d_model : dimension of the model , number of features or dimensions that each element in input sequence should have
        max_seq_len : maximum size of caption that decoder can handle
    """
    def __init__(self, d_model, max_seq_len = 80):
        super().__init__()
        self.d_model = d_model
        
        # create constant 'pe' matrix with values dependant on 
        # pos and i
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len): # itereates over the maximum length of sequence and compute pe for each position
            for i in range(0, d_model, 2):
                pe[pos, i] = np.sin(pos / (10000 ** ((2 * i)/d_model)))
                pe[pos, i + 1] = np.cos(pos / (10000 ** ((2 * (i + 1))/d_model)))
                
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe) # register pe as a buffer which means its parameters saves when model parameters saves

 
    
    def forward(self, x):
        # make embeddings relatively larger
        x = x * np.sqrt(self.d_model)
        #add constant to embedding
        seq_len = x.size(1) # gets the length of the sequence
        x = x + Variable(self.pe[:,:seq_len], \
        requires_grad=False).cuda()
        return x
    
def mask_handling():
    pass

=======
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable

class ScaleDotProduct(nn.Module):
    def __init__(self,query_dim,key_dim,value_dim):
        super(ScaleDotProduct,self).__init__()
        self.scale = nn.Parameter(torch.Tensor([np.sqrt(key_dim)]))
        
    def forward(self, mask, query, keys, values):
            # query : [B,Q] (hidden_state,decoder_output)
            # Keys : [T,B,K] (encoder outputs)
            # Values : [T,B,V] (encoder outputs)
            # assume Q == K

            # compute energy:
            query = query.unsqueeze(1) # [B,Q] ==> [B,1,Q]
            print(f"Query shape is {query.shape}")
            keys = keys.permute(1,2,0) # [T,B,K] ==> [B,K,T]
            print(f"Keys shape is {keys.shape}")
            energy = torch.bmm(query,keys) # [B,1,Q] * [B,K,T] ==> [B,1,T]
            energy = torch.multiply(self.scale, energy)  # apply scale
            print(f"energy1 shape is {energy.shape}")
            energy = F.softmax(energy,dim=2)
            print(f"energy2 shape is {energy.shape}")

            # apply mask and renormalize
            energy = energy * mask
            energy = energy.div(energy.sum(2,keepdim=True))
            print(f"energy3 shape is {energy.shape}")

            # weight values
            values = values.transpose(0,1) # [T,B,V] -> [B,T,V]
            combo = torch.bmm(energy, values).squeeze(1) # [B,1,T]*[B,T,V] -> [B,V]
            print(f"values shape is {values.shape}")
            print(f"combo shape is {combo.shape}")

            return (combo, energy)



class AdditiveAttention(nn.Module):
    def __init__(self,input_size,hidden_size,output_size,dropout=0.5):
        super(AdditiveAttention,self).__init__()
        self.query = nn.Linear(input_size,hidden_size)
        # self.bias = nn.Parameter()
        self.key = nn.Linear(input_size,hidden_size)
        self.value = nn.Linear(input_size,output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self,inp):
        query = self.query(inp)
        key = self.key(inp)
        attn_weight = torch.matmul(query,key.T)
        attn_weight = F.softmax(attn_weight)
        attn_weight = self.dropout(attn_weight)
        output = self.value(attn_weight)
        return output
    


class PositionalEncoder(nn.Module):
    """
        d_model : dimension of the model , number of features or dimensions that each element in input sequence should have
        max_seq_len : maximum size of caption that decoder can handle
    """
    def __init__(self, d_model, max_seq_len = 80):
        super().__init__()
        self.d_model = d_model
        
        # create constant 'pe' matrix with values dependant on 
        # pos and i
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len): # itereates over the maximum length of sequence and compute pe for each position
            for i in range(0, d_model, 2):
                pe[pos, i] = np.sin(pos / (10000 ** ((2 * i)/d_model)))
                pe[pos, i + 1] = np.cos(pos / (10000 ** ((2 * (i + 1))/d_model)))
                
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe) # register pe as a buffer which means its parameters saves when model parameters saves

 
    
    def forward(self, x):
        # make embeddings relatively larger
        x = x * np.sqrt(self.d_model)
        #add constant to embedding
        seq_len = x.size(1) # gets the length of the sequence
        x = x + Variable(self.pe[:,:seq_len], \
        requires_grad=False).cuda()
        return x
    
def mask_handling():
    pass

>>>>>>> f72c2826b6880f5d1c126923f46f030081e3e866
