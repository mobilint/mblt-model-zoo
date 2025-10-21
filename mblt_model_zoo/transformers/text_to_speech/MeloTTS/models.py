import math
import torch
from torch import nn

from melo import commons

import maccel
from maccel import Cluster, Core, CoreId
import numpy as np

from transformers.utils import logging


logger = logging.get_logger(__name__)


class MobilintTextEncoderAndDurationPredictor(nn.Module):
    def __init__(
        self,
        
        n_vocab,
        hidden_channels,
        num_languages=None,
        num_tones=None,
        
        mxq_path="",
    ):
        super().__init__()
        
        if num_languages is None:
            from melo.text import num_languages
        if num_tones is None:
            from melo.text import num_tones
        self.n_vocab = n_vocab
        self.hidden_channels = hidden_channels
        self.emb = nn.Embedding(n_vocab, hidden_channels)
        nn.init.normal_(self.emb.weight, 0.0, hidden_channels**-0.5)
        self.tone_emb = nn.Embedding(num_tones, hidden_channels)
        nn.init.normal_(self.tone_emb.weight, 0.0, hidden_channels**-0.5)
        self.language_emb = nn.Embedding(num_languages, hidden_channels)
        nn.init.normal_(self.language_emb.weight, 0.0, hidden_channels**-0.5)
        
        self.acc = maccel.Accelerator()
        mc = maccel.ModelConfig()
        mc.set_single_core_mode(core_ids=[CoreId(Cluster.Cluster0, Core.Core0)])
        self.mxq_model = maccel.Model(mxq_path, mc)
        num_model_variants = self.mxq_model.get_num_model_variants()
        self.allowed_chunks = [
            self.mxq_model.get_model_variant_handle(i).get_model_input_shape()[0][1]
            for i in range(num_model_variants)
        ]
        self.mxq_model.launch(self.acc)
    
    def __call__(
        self,
        x,
        x_lengths,
        tone,
        language,
        ja_bert,
        noise_scale,
    ):
        # b = batch size, h = hidden_channels, t = time (dynamic dim)
        # ja_bert [1, f, t]
        x = (
            self.emb(x)
            + self.tone_emb(tone)
            + self.language_emb(language)
        ) # [b, t, h]
        x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.shape[1]), 1).to(
            x.dtype
        ) # [t, 1]
        
        z = (
            torch.randn(x.shape[0], 2, x.shape[1]).to(device=x.device, dtype=x.dtype)
            * noise_scale
        ) # [b, 2, t]
        z0, z1 = torch.split(z, [1, 1], 1) # [b, 1, t], [b, 1, t]
        
        x = x.unsqueeze(1).type(torch.float32).cpu().numpy() # [b, 1, t, h]
        ja_bert = ja_bert.permute(0, 2, 1).unsqueeze(1).type(torch.float32).cpu().numpy() # [b, 1, t, f]
        z0 = z0.permute(0, 2, 1).unsqueeze(1).type(torch.float32).cpu().numpy() # [b, 1, t, 1]
        z1 = z1.permute(0, 2, 1).unsqueeze(1).type(torch.float32).cpu().numpy() # [b, 1, t, 1]
        
        max_chunk = max(self.allowed_chunks)
        num_of_chunks = math.ceil(x.shape[2] / max_chunk)
        
        m_p_chunks, logs_p_chunks, logw_chunks = [], [], []
        
        print(self.allowed_chunks, x.shape, ja_bert.shape, z0.shape, z1.shape)
        
        for i in range(num_of_chunks):
            start_index = i * max_chunk
            end_index = start_index + max_chunk
            remaining_length = x.shape[2] - start_index
            
            if end_index > x.shape[2]:
                chunk_size = min([chunk_size for chunk_size in self.allowed_chunks if chunk_size >= remaining_length])
                pad_width = [(0, 0), (0, 0), (0, chunk_size - remaining_length), (0, 0)]
                
                x_slice = np.pad(x[:, :, start_index:, :], pad_width, mode="constant", constant_values=0)
                ja_bert_slice = np.pad(ja_bert[:, :, start_index:, :], pad_width, mode="constant", constant_values=0)
                z0_slice = np.pad(z0[:, :, start_index:, :], pad_width, mode="constant", constant_values=0)
                z1_slice = np.pad(z1[:, :, start_index:, :], pad_width, mode="constant", constant_values=0)
            else:
                x_slice = x[:, :, start_index:end_index, :]
                ja_bert_slice = ja_bert[:, :, start_index:end_index, :]
                z0_slice = z0[:, :, start_index:end_index, :]
                z1_slice = z1[:, :, start_index:end_index, :]
            
            print(start_index, end_index, remaining_length, z1_slice.shape, x_slice.shape, ja_bert_slice.shape, z0_slice.shape)

            m_p_chunk, logs_p_chunk, logw_chunk = self.mxq_model.infer([z1_slice, x_slice, ja_bert_slice, z0_slice])
            
            if end_index > x.shape[2]:
                m_p_chunk = m_p_chunk[..., :remaining_length, :]
                logs_p_chunk = logs_p_chunk[..., :remaining_length, :]
                logw_chunk = logw_chunk[..., :remaining_length]
            
            m_p_chunks.append(m_p_chunk)
            logs_p_chunks.append(logs_p_chunk)
            logw_chunks.append(logw_chunk)
        
        # Concatenate along sequence axis
        m_p = np.concatenate(m_p_chunks, axis=2)
        logs_p = np.concatenate(logs_p_chunks, axis=2)
        logw = np.concatenate(logw_chunks, axis=2)
        
        # Convert to torch tensors
        m_p = torch.tensor(m_p, dtype=torch.float32, device=self.device).squeeze(1).transpose(1, 2)
        logs_p = torch.tensor(logs_p, dtype=torch.float32, device=self.device).squeeze(1).transpose(1, 2)
        logw = torch.tensor(logw, dtype=torch.float32, device=self.device)
        
        return m_p, logs_p, x_mask, logw

class MobilintTransformerCouplingBlockAndGenerator(nn.Module):
    def __init__(
        self,
        channels,
        mxq_path,
    ):
        assert channels % 2 == 0, "channels should be divisible by 2"
        super().__init__()
        self.channels = channels
        self.half_channels = channels // 2
        
        self.acc = maccel.Accelerator()
        mc = maccel.ModelConfig()
        mc.set_single_core_mode(core_ids=[CoreId(Cluster.Cluster0, Core.Core2)])
        self.mxq_model = maccel.Model(mxq_path, mc)
        num_model_variants = self.mxq_model.get_num_model_variants()
        self.allowed_chunks = [
            self.mxq_model.get_model_variant_handle(i).get_model_input_shape()[0][1]
            for i in range(num_model_variants)
        ]
        self.mxq_model.launch(self.acc)
        
    def __call__(self, x):
        x = x.permute(0, 2, 1).unsqueeze(1).type(torch.float32).cpu().numpy() # (1, C, W) -> (1, W, C) -> (1, 1, W, C)
        x = np.ascontiguousarray(x.to("cpu").numpy())
        
        max_chunk = max(self.allowed_chunks)
        num_of_chunks = math.ceil(x.shape[2] / max_chunk)
        
        output_chunks = []
        
        for i in range(num_of_chunks):
            start_index = i * max_chunk
            end_index = start_index + max_chunk
            remaining_length = x.shape[2] - start_index
            
            if end_index > x.shape[2]:
                chunk_size = min([chunk_size for chunk_size in self.allowed_chunks if chunk_size >= remaining_length])
                pad_width = [(0, 0), (0, 0), (0, chunk_size - remaining_length), (0, 0)]
                
                x_slice = np.pad(x[:, :, start_index:, :], pad_width, mode="constant", constant_values=0)
            else:
                x_slice = x[:, :, start_index:end_index, :]
            
            x0, x1 = torch.split(x_slice, [self.half_channels, self.half_channels], 2) # [1, 1, W, C // 2], [1, 1, W, C // 2]
            x0 = x0.flip([3])
            
            output_chunk = self.mxq_model.infer([x1, x0])[0] # (1, seq_len, 1)
            
            if end_index > x.shape[2]:
                output_chunk = output_chunk[:, :(remaining_length * self.upsample_initial_channel), :]
            
            output_chunks.append(output_chunk)

        output = np.concatenate(output_chunks, 1) # (1, seq_len, 1)
        output = torch.tensor(output, dtype=torch.float32, device=self.device).squeeze(2).unsqueeze(0)  # (1, 1, seq_len)

        return None, output

class MobilintSynthesizerTrn(nn.Module):
    def __init__(
        self,
        n_vocab,
        spec_channels,
        segment_size,
        inter_channels,
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers,
        kernel_size,
        p_dropout,
        resblock,
        resblock_kernel_sizes,
        resblock_dilation_sizes,
        upsample_rates,
        upsample_initial_channel,
        upsample_kernel_sizes,
        n_speakers=256,
        gin_channels=256,
        use_sdp=True,
        n_flow_layer=4,
        n_layers_trans_flow=6,
        flow_share_parameter=False,
        use_transformer_flow=True,
        use_vc=False,
        num_languages=None,
        num_tones=None,
        norm_refenc=False,
        mxq_path_enc_p_sdp_dp="",
        mxq_path_dec_flow="",
        **kwargs
    ):
        super().__init__()
        
        # self.enc_p, self.dp, self.sdp all in one mxq
        self.enc_p_sdp_dp = MobilintTextEncoderAndDurationPredictor(
            n_vocab,
            hidden_channels,
            num_languages=num_languages,
            num_tones=num_tones,
            mxq_path=mxq_path_enc_p_sdp_dp,
        )
        
        # self.dec, self.flow all in one mxq
        self.dec_flow = MobilintTransformerCouplingBlockAndGenerator(
            inter_channels,
            mxq_path=mxq_path_dec_flow,
        )
        
        if n_speakers <= 0:
            logger.warning("`self.n_speakers` should be positive to use g with self.emb_g.")
        
        if use_vc is True:
            logger.warning("`self.use_vc` should be False to use g_p as g.")

    def infer(
        self,
        x,
        x_lengths,
        sid,
        tone,
        language,
        bert,
        ja_bert,
        noise_scale=0.667,
        length_scale=1,
        noise_scale_w=0.8,
        max_len=None,
        sdp_ratio=0,
        y=None,
        g=None,
    ):
        if x_lengths.shape != [1] or x_lengths[0] != int(x.shape[1]):
            logger.warning_once(f"Input `x_lengths` is set to `[x.shape[1]]` inside the mxq. x_length.shape={x_lengths.shape}, x_lengths[0]={x_lengths[0]}, x.shape={x.shape}")
            
        if sid != 0:
            logger.warning_once("Input `sid` is set to 0 inside the mxq.")
        
        if g is not None:
            logger.warning_once('Input `g` is calculated inside the mxq with assuming sid is 0.')
                
        if sdp_ratio != 0.2:
            logger.warning_once('Input `sdp_ratio` is set inside the mxq as 0.2.')
        
        if max_len is not None:
            logger.warning_once('Input `max_len` is not supported.')
        
        m_p, logs_p, x_mask, logw = self.enc_p_sdp_dp(
            x,
            x_lengths,
            tone,
            language,
            ja_bert,
            noise_scale_w,
        )
        w = torch.exp(logw) * x_mask * length_scale
        
        w_ceil = torch.ceil(w)
        y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
        y_mask = torch.unsqueeze(commons.sequence_mask(y_lengths, None), 1).to(
            x_mask.dtype
        )
        attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
        attn = commons.generate_path(w_ceil, attn_mask)

        m_p = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(
            1, 2
        )  # [b, t', t], [b, t, d] -> [b, d, t']
        logs_p = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(
            1, 2
        )  # [b, t', t], [b, t, d] -> [b, d, t']

        z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale
        z, o = self.flow_dec(z_p)
        # print('max/min of o:', o.max(), o.min())
        return o, attn, y_mask, (z, z_p, m_p, logs_p)
