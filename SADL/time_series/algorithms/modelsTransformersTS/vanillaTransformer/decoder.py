import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .attn import MultiHeadAttention
from .embed import PositionalEmbedding,PositionalEncoding

class DecoderLayer(nn.Module):
  def __init__(self, feedforward_hunits, n_heads, d_model,d_keys,d_values,dropout_rate = 0.1) -> None:
       super(DecoderLayer, self).__init__()
       self.feedforward_hunits = feedforward_hunits
       self.n_heads = n_heads
       self.dropout = dropout_rate
       self.d_model = d_model
       self.d_k = d_keys
       self.d_v = d_values
       self.build_model()

  def build_model(self):

     #Define Multi_head_attetion, causal attention
       self.multi_headattention = MultiHeadAttention(self.d_model, self.n_heads,  self.d_k, self.d_v,self.dropout)

     # Define  Multi_head_attetion, encoder-decoder attention
       self.multi_headattention_enc_dec = MultiHeadAttention(self.d_model, self.n_heads,  self.d_k, self.d_v,self.dropout)

     # Fully connected feed forward layer,position-wise
       self.ff_1 = nn.Linear(self.d_model, self.feedforward_hunits)
       self.ff_2 = nn.Linear(self.feedforward_hunits, self.d_model)

     # Layer normalization and dropout
       self.dropout = nn.Dropout(self.dropout)
       self.norm = nn.LayerNorm(self.d_model, eps=1e-6)


  def forward(self,decoder_in,encoder_out, mask = None, mask_enc_dec = None):
      # causal attention and encoder-decoder attention
      attn_decout, attn = self.multi_headattention(decoder_in,decoder_in,decoder_in,mask)
      attn_decout, attn_enc_dec = self.multi_headattention_enc_dec (attn_decout,encoder_out,encoder_out,mask_enc_dec)

      out = self.ff_1(attn_decout)
      out = self.ff_2(F.relu(out))

      # residual connection and layer normalization
      out = self.dropout(out)
      out = self.norm(out + attn_decout)

      return out,attn,attn_enc_dec




class Decoder(nn.Module):
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
        super(Decoder, self).__init__()
        self.n_layers = n_layers
        self.d_model = d_model
        self.feedforward_units = feedforward_units
        self.n_heads = n_heads
        self.norm = nn.LayerNorm(d_model, eps=1e-6)
        self.scale = embedding_scale
        self.input_size = input_size
        self.seq_len = seq_len
        
        
        # Input projection (instead of embedding) for real-valued inputs
        self.input_projection = nn.Linear(self.input_size, self.d_model)

        # New positional embedding
        #self.pos_embedding = PositionalEmbedding(d_model)  
        self.pos_embedding = PositionalEncoding( self.d_model, 0.1,self.seq_len)

        self.dropout = nn.Dropout(dropout_rate)

        # Stack of DecoderLayers
        self.dec_layers = nn.ModuleList([
            DecoderLayer(
                self.feedforward_units,
                self.n_heads,
                self.d_model,
                d_keys,
                d_values,
                dropout_rate
            ) for _ in range(n_layers)
        ])

    def forward(self, inputs, encoder_out, mask_attn, mask_enc_dec):
        """
        inputs: [batch_size, tgt_len, input_size]
        encoder_out: [batch_size, src_len, d_model]
        """
        attn_list = []
        attn_enc_dec_list = []

        # Project input to model dimension
        out = self.input_projection(inputs)  # [B, L, d_model]

        if self.scale:
            out *= math.sqrt(self.d_model)

        # Add positional embedding
        out = out + self.pos_embedding(out)
        out = self.norm(self.dropout(out))

        # Decoder layers
        for i in range(self.n_layers):
            out, attn, attn_enc_dec = self.dec_layers[i](out, encoder_out, mask_attn, mask_enc_dec)
            attn_list.append(attn)
            attn_enc_dec_list.append(attn_enc_dec)

        return out, attn_list, attn_enc_dec_list