import torch
import torch.nn as nn
import torch.nn.functional as F
from .attn import MultiHeadAttention
from .embed import PositionalEmbedding,PositionalEncoding
import math


class EncoderLayer(nn.Module):
  def __init__(self, feedforward_hunits, n_heads, d_model,d_keys,d_values,dropout_rate = 0.1):
       super(EncoderLayer, self).__init__()
       self.feedforward_hunits = feedforward_hunits
       self.n_heads = n_heads
       self.dropout = dropout_rate
       self.d_model = d_model
       self.d_k = d_keys
       self.d_v = d_values
       self.build_model()

  def build_model(self):

     #Define Multi_head_attetion
      self.multi_headattention = MultiHeadAttention(self.d_model, self.n_heads,  self.d_k, self.d_v,self.dropout)

     # Fully connected feed forward layer,position-wise
      self.ff_1 = nn.Linear(self.d_model, self.feedforward_hunits)
      self.ff_2 = nn.Linear(self.feedforward_hunits, self.d_model)

     # Layer normalization and dropout
      self.dropout = nn.Dropout(self.dropout)
      self.norm = nn.LayerNorm(self.d_model, eps=1e-6)


  def forward(self,encoder_in,mask = None):
      attn_enout,attn = self.multi_headattention(encoder_in,encoder_in,encoder_in,mask)

      out = self.ff_1(attn_enout)
      out = self.ff_2(F.relu(out))

      # residual connection and layer normalization
      out = self.dropout(out)
      out = self.norm(out + attn_enout)

      return out,attn




class Encoder(nn.Module):
    def __init__(self,
                 n_layers,
                 feedforward_units,
                 seq_len,
                 n_heads,
                 input_size,
                 d_model,
                 d_keys,
                 d_values,
                 dropout_rate=0.1,
                 embedding_scale=False):
        super(Encoder, self).__init__()
        self.n_layers = n_layers
        self.d_model = d_model
        self.feedforward_units = feedforward_units
        self.n_heads = n_heads
        self.norm = nn.LayerNorm(d_model, eps=1e-6)
        self.scale = embedding_scale
        self.input_size = input_size
        self.seq_len = seq_len
         
        self.input_projection = nn.Linear(self.input_size, self.d_model)
        #self.pos_embedding = PositionalEmbedding(d_model)  
        self.pos_embedding = PositionalEncoding( self.d_model, 0.1,seq_len)
        self.dropout = nn.Dropout(dropout_rate)

        self.enc_layers = nn.ModuleList([
            EncoderLayer(
                self.feedforward_units,
                n_heads,
                d_model,
                d_keys,
                d_values,
                dropout_rate
            ) for _ in range(n_layers)
        ])

    def forward(self, inputs, mask):
        """
        inputs: [batch_size, seq_len, input_size]
        mask: [batch_size, 1, 1, seq_len]
        """
        attn_list = []
        
        out = self.input_projection(inputs)  # [B, L, d_model]
    
        if self.scale:
            out *= math.sqrt(self.d_model)
       
        out = out + self.pos_embedding(out)  
        out = self.norm(self.dropout(out))

        for i in range(self.n_layers):
            out, attn = self.enc_layers[i](out, mask=mask)
            attn_list.append(attn)

        return out, attn_list