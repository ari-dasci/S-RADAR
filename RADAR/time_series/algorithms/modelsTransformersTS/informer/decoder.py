import torch
import torch.nn as nn
import torch.nn.functional as F

class DecoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
                 dropout=0.1, activation="relu"):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4*d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        
        # Self-Attention y almacenar la matriz de atención
        self_attn_output, self_attn_weights = self.self_attention(
            x, x, x,
            attn_mask=x_mask
        )
        x = x + self.dropout(self_attn_output)
        x = self.norm1(x)

         # Cross-Attention y almacenar la matriz de atención
        cross_attn_output, cross_attn_weights = self.cross_attention(
            x, cross, cross,
            attn_mask=cross_mask
        )
        x = x + self.dropout(cross_attn_output)
        y = x = self.norm2(x)
        # Feedforward y normalización
        y = self.dropout(self.activation(self.conv1(y.transpose(-1,1))))
        y = self.dropout(self.conv2(y).transpose(-1,1))

        return self.norm3(x+y), self_attn_weights, cross_attn_weights

class Decoder(nn.Module):
    def __init__(self, layers, norm_layer=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        self_attentions = []  
        cross_attentions = []  
        for layer in self.layers:
            x, self_attn, cross_attn = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)
            self_attentions.append(self_attn)
            cross_attentions.append(cross_attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, self_attentions, cross_attentions