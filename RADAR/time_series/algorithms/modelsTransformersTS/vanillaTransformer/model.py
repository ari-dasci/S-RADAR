import torch
import torch.nn as nn
import torch.nn.functional as F
from .encoder import Encoder
from .decoder import Decoder


import torch
import torch.nn as nn

class Transformer(nn.Module):
    """
    Transformer para detección de anomalías en series temporales.

    Args:
        size_enc_in: dimensión de entrada del encoder
        size_dec_in: dimensión de entrada/salida del decoder
        ulayers_feedfwd: número de unidades en las capas feedforward
        d_qk: dimensión de claves y consultas (Q/K)
        d_v: dimensión de los valores (V)
        d_model: dimensión interna del modelo
        n_layers: número de capas en encoder y decoder
        n_heads: número de cabezas de atención
        dropout_rate: tasa de dropout
        embedding_scale: si escalar embeddings por sqrt(d_model)
        attns_outs: si retornar atenciones como salida
    """
    def __init__(self,
                 size_enc_in,
                 size_dec_in,
                 ulayers_feedfwd,
                 seq_len,
                 d_qk=64,
                 d_v=64,
                 d_model=512,
                 n_layers=6,
                 n_heads=8,
                 dropout_rate=0.1,
                 embedding_scale=False,
                 attns_outs=False):
        super(Transformer, self).__init__()
        self.attns_outs = attns_outs

        self.encoder = Encoder(
            n_layers=n_layers,
            feedforward_units=ulayers_feedfwd,
            seq_len = seq_len,
            n_heads=n_heads,
            input_size=size_enc_in,
            d_model=d_model,
            d_keys=d_qk,
            d_values=d_v,
            dropout_rate=dropout_rate,
            embedding_scale=embedding_scale
        )

        self.decoder = Decoder(
            n_layers=n_layers,
            feedforward_units=ulayers_feedfwd,
            seq_len = seq_len,
            n_heads=n_heads,
            input_size=size_dec_in,
            d_model=d_model,
            d_keys=d_qk,
            d_values=d_v,
            dropout_rate=dropout_rate,
            embedding_scale=embedding_scale
        )

        self.linear = nn.Linear(d_model, size_dec_in, bias=False)

    def gen_mask(self, src, tgt):
        """
        Genera máscaras para atención en series temporales.

        src: [B, L_src, D]
        tgt: [B, L_tgt, D]
        """
        # Máscara de padding (asume padding con ceros)
        src_mask = (src.abs().sum(dim=-1) != 0).unsqueeze(1).unsqueeze(2)  # [B,1,1,L_src]
        tgt_mask = (tgt.abs().sum(dim=-1) != 0).unsqueeze(1).unsqueeze(2)  # [B,1,1,L_tgt]

        # Máscara para evitar mirar hacia el futuro en decoder
        seq_len = tgt.size(1)
        nopeak_mask = torch.tril(torch.ones(1, 1, seq_len, seq_len, device=tgt.device)).bool()

        tgt_mask = tgt_mask & nopeak_mask  # [B,1,L,L]
        return src_mask, tgt_mask

    def forward(self, enc_inputs, dec_inputs):
        """
        enc_inputs: [B, L_enc, D_enc]
        dec_inputs: [B, L_dec, D_dec]
        """
        mask_enc, mask_dec = self.gen_mask(enc_inputs, dec_inputs)

        # Paso por el encoder
        enc_out, attns_enc = self.encoder(enc_inputs, mask_enc)

        # Paso por el decoder
        dec_out, attns_dec, attns_enc_dec = self.decoder(
            dec_inputs, enc_out, mask_dec, mask_enc
        )

        # Proyección final
        out = self.linear(dec_out)  # [B, L_dec, size_dec_in]

        if self.attns_outs:
            return out, attns_enc, attns_dec, attns_enc_dec

        return out