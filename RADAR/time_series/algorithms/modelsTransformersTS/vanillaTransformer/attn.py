import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, d_keys = None, dropout=0.1):
        super().__init__()
        self.d_k = d_keys
        self.dropout = nn.Dropout(dropout)


    def forward(self, queries, keys, values, mask=None):
         mul = torch.matmul(queries,keys.transpose(-2,-1))
         d_k = keys.size(-1) if self.d_k is None else self.d_k                     # Convert to float ? *********
         scale = mul / sqrt(d_k)

        # Apply mask
         if mask is not None:
             scale += (mask * -1e9)

         a = self.dropout(F.softmax(scale, dim=-1))
         out = torch.matmul(a, values)
         return out,a



class MultiHeadAttention(nn.Module):
    '''
      The MultiHeadAttention class is used to perform multi-head attention in
      the transformer model. It splits the input vectors into different heads or
      projections, calculates the attention scores, and concatenates the results
      before applying a linear transformation to obtain the final output.

       d_model: number of units in the model (dimensionality of the feature vectors)
    '''
    def __init__(self,d_model, n_heads,
                 d_keys=None, d_values=None,dropout_rate=0.1):

        super().__init__()
        self.d_model = d_model
        self.d_keys = d_keys
        self.d_values = d_values
        self.n_heads = n_heads
        self.dropout_rate = nn.Dropout(dropout_rate)
        self.build_model()


    def build_model(self):
        assert self.d_model % self.n_heads == 0
        self.d_head = self.d_model // self.n_heads                                  # dimension of every headi ****
        # weight matrices for Q, K,V and output W0
        self.qw = nn.Linear(self.d_model, self.n_heads * self.d_keys, bias=False)
        self.kw = nn.Linear(self.d_model, self.n_heads * self.d_keys, bias=False)
        self.vw = nn.Linear(self.d_model, self.n_heads * self.d_values, bias=False)
        self.fw = nn.Linear(self.n_heads * self.d_values, self.d_model, bias=False)
        self.norm = nn.LayerNorm(self.d_model, eps=1e-6)
        self.attn = ScaledDotProductAttention( d_keys = self.d_keys)


    def forward(self, queries, keys, values, mask):
        residual = queries
        nh = self.n_heads
        batch_size = queries.size(0)
        qsize = queries.size(1)
        # Query, Key and Value matrices
        queries = (self.qw(queries).view(batch_size, qsize, nh, self.d_keys)).transpose(1,2)
        keys = (self.kw(keys).view(batch_size, keys.size(1), nh, self.d_keys)).transpose(1,2)
        values = (self.vw(values).view(batch_size, values.size(1), nh, self.d_values)).transpose(1,2)

        # Get the attention scores  and concat
        output, attention = self.attn(queries, keys, values, mask)

        output = output.transpose(2,1).contiguous().view(batch_size ,qsize, -1)

        #get the output
        outputs = self.dropout_rate(self.fw(output))                          # dropout and residual here or elayer y dlayer ?  *******

        #Normalize
        outputs = self.norm(outputs + residual)

        return outputs,attention

