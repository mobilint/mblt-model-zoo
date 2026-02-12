import math
from typing import List, Literal, Optional, Union

import numpy as np
import qbruntime
import torch
from qbruntime import Cluster, CoreId
from torch import nn
from transformers.utils import logging

from mblt_model_zoo.utils.npu_backend import MobilintNPUBackend

from . import commons

logger = logging.get_logger(__name__)


class MobilintTextEncoderAndDurationPredictor(nn.Module):
    def __init__(
        self,
        n_vocab,
        hidden_channels,
        num_languages=None,
        num_tones=None,
        
        name_or_path: str = "",
        mxq_path: str = "",
        dev_no: int = 0,
        core_mode: Literal["single", "multi", "global4", "global8"] = "single",
        target_cores: Optional[List[Union[str, "CoreId"]]] = None,
        target_clusters: Optional[List[Union[int, "Cluster"]]] = None,
        no_launch: bool = False,
    ):
        super().__init__()

        if num_languages is None:
            from .text import num_languages
        if num_tones is None:
            from .text import num_tones
        self.n_vocab = n_vocab
        self.hidden_channels = hidden_channels
        self.emb = nn.Embedding(n_vocab, hidden_channels)
        nn.init.normal_(self.emb.weight, 0.0, hidden_channels**-0.5)
        self.tone_emb = nn.Embedding(num_tones, hidden_channels)
        nn.init.normal_(self.tone_emb.weight, 0.0, hidden_channels**-0.5)
        self.language_emb = nn.Embedding(num_languages, hidden_channels)
        nn.init.normal_(self.language_emb.weight, 0.0, hidden_channels**-0.5)
        
        self.npu_backend = MobilintNPUBackend(
            mxq_path=mxq_path,
            dev_no=dev_no,
            core_mode=core_mode,
            target_cores=target_cores,
            target_clusters=target_clusters,
        )
        
        self.npu_backend.name_or_path = name_or_path
        self.npu_backend.create()
        if no_launch != True:
            self.npu_backend.launch()
        
        num_model_variants = self.npu_backend.mxq_model.get_num_model_variants()
        self.allowed_chunks = [
            self.npu_backend.mxq_model.get_model_variant_handle(i).get_model_input_shape()[0][1]
            for i in range(num_model_variants)
        ]
    
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
        x = self.emb(x) + self.tone_emb(tone) + self.language_emb(language)  # [b, t, h]
        x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.shape[1]), 1).to(
            x.dtype
        )  # [t, 1]

        z = (
            torch.randn(x.shape[0], 2, x.shape[1]).to(device=x.device, dtype=x.dtype)
            * noise_scale
        ) # [b, 2, t]
        z0, z1 = torch.split(z, [1, 1], 1) # [b, 1, t], [b, 1, t]
        
        x = x.type(torch.float32).cpu().numpy() # [b, t, h]
        ja_bert = ja_bert.permute(0, 2, 1).type(torch.float32).cpu().numpy() # [b, t, f]
        z0 = z0.permute(0, 2, 1).type(torch.float32).cpu().numpy() # [b, t, 1]
        z1 = z1.permute(0, 2, 1).type(torch.float32).cpu().numpy() # [b, t, 1]

        max_chunk = max(self.allowed_chunks)
        num_of_chunks = math.ceil(x.shape[1] / max_chunk)

        m_p_chunks, logs_p_chunks, logw_chunks = [], [], []

        for i in range(num_of_chunks):
            start_index = i * max_chunk
            end_index = start_index + max_chunk
            remaining_length = x.shape[1] - start_index

            if end_index > x.shape[1]:
                chunk_size = min(
                    [
                        chunk_size
                        for chunk_size in self.allowed_chunks
                        if chunk_size >= remaining_length
                    ]
                )
                pad_width = [(0, 0), (0, chunk_size - remaining_length), (0, 0)]

                x_slice = np.pad(
                    x[:, start_index:, :], pad_width, mode="constant", constant_values=0
                )
                ja_bert_slice = np.pad(
                    ja_bert[:, start_index:, :],
                    pad_width,
                    mode="constant",
                    constant_values=0,
                )
                z0_slice = np.pad(
                    z0[:, start_index:, :],
                    pad_width,
                    mode="constant",
                    constant_values=0,
                )
                z1_slice = np.pad(
                    z1[:, start_index:, :],
                    pad_width,
                    mode="constant",
                    constant_values=0,
                )
            else:
                x_slice = x[:, start_index:end_index, :]
                ja_bert_slice = ja_bert[:, start_index:end_index, :]
                z0_slice = z0[:, start_index:end_index, :]
                z1_slice = z1[:, start_index:end_index, :]
            
            input_mask = np.ones(shape=(2, x_slice.shape[1], x_slice.shape[1]), dtype=np.float32)
            input_mask[:, :, :remaining_length] = 0
            
            outputs = self.npu_backend.mxq_model.infer([ja_bert_slice, z1_slice, z0_slice, x_slice, input_mask])
            assert outputs is not None, "No output from mxq model inference."
            logw_chunk, m_p_chunk, logs_p_chunk = outputs

            if end_index > x.shape[1]:
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
        m_p = (
            torch.tensor(m_p, dtype=torch.float32, device=x_lengths.device)
            .squeeze(1)
            .transpose(1, 2)
        )
        logs_p = (
            torch.tensor(logs_p, dtype=torch.float32, device=x_lengths.device)
            .squeeze(1)
            .transpose(1, 2)
        )
        logw = torch.tensor(logw, dtype=torch.float32, device=x_lengths.device)

        return m_p, logs_p, x_mask, logw

    def launch(self):
        self.npu_backend.launch()
    
    def dispose(self):
        self.npu_backend.dispose()

class MobilintTransformerCouplingBlockAndGenerator(nn.Module):
    def __init__(
        self,
        channels,
        upsample_initial_channel,
        
        name_or_path: str = "",
        mxq_path: str = "",
        dev_no: int = 0,
        core_mode: Literal["single", "multi", "global4", "global8"] = "single",
        target_cores: Optional[List[Union[str, "CoreId"]]] = None,
        target_clusters: Optional[List[Union[int, "Cluster"]]] = None,
        no_launch: bool = False,
    ):
        assert channels % 2 == 0, "channels should be divisible by 2"
        super().__init__()
        self.channels = channels
        self.half_channels = channels // 2
        self.upsample_initial_channel = upsample_initial_channel
        
        self.npu_backend = MobilintNPUBackend(
            mxq_path=mxq_path,
            dev_no=dev_no,
            core_mode=core_mode,
            target_cores=target_cores,
            target_clusters=target_clusters,
        )
        
        self.npu_backend.name_or_path = name_or_path
        self.npu_backend.create()
        if no_launch != True:
            self.npu_backend.launch()
        
        num_model_variants = self.npu_backend.mxq_model.get_num_model_variants()
        self.allowed_chunks = [
            self.npu_backend.mxq_model.get_model_variant_handle(i).get_model_input_shape()[0][1]
            for i in range(num_model_variants)
        ]
        
    def __call__(self, x):
        device = x.device
        x = (
            x.permute(0, 2, 1).type(torch.float32).cpu().numpy()
        )  # (1, C, W) -> (1, W, C)
        x = np.ascontiguousarray(x)

        max_chunk = max(self.allowed_chunks)
        num_of_chunks = math.ceil(x.shape[1] / max_chunk)

        output_chunks = []

        for i in range(num_of_chunks):
            start_index = i * max_chunk
            end_index = start_index + max_chunk
            remaining_length = x.shape[1] - start_index

            if end_index > x.shape[1]:
                chunk_size = min(
                    [
                        chunk_size
                        for chunk_size in self.allowed_chunks
                        if chunk_size >= remaining_length
                    ]
                )
                pad_width = [(0, 0), (0, chunk_size - remaining_length), (0, 0)]

                x_slice = np.pad(
                    x[:, start_index:, :], pad_width, mode="constant", constant_values=0
                )
            else:
                x_slice = x[:, start_index:end_index, :]

            x_slice = np.split(
                x_slice, [self.half_channels], 2
            )  # [1, W, C // 2], [1, W, C // 2]
            x0, x1 = x_slice[0], x_slice[1]
            x0 = np.flip(x0, 2)

            input_mask = np.ones(shape=(2, x0.shape[1], x0.shape[1]), dtype=np.float32)
            input_mask[:, :, :remaining_length] = 0
            
            outputs = self.npu_backend.mxq_model.infer([x1, x0, input_mask]) # [(1, seq_len, 1)]
            assert outputs is not None, "No output from mxq model inference."
            output_chunk = outputs[0]

            if end_index > x.shape[1]:
                output_chunk = output_chunk[
                    :, : (remaining_length * self.upsample_initial_channel), :
                ]

            output_chunks.append(output_chunk)

        output = np.concatenate(output_chunks, 1)  # (1, seq_len, 1)
        output = (
            torch.tensor(output, dtype=torch.float32, device=device)
            .squeeze(2)
            .unsqueeze(0)
        )  # (1, 1, seq_len)

        return None, output

    def launch(self):
        self.npu_backend.launch()
    
    def dispose(self):
        self.npu_backend.dispose()

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
        
        name_or_path="",
        dev_no=0,
        target_core="0:0",
        encoder_mxq_path="",
        decoder_mxq_path="",
        
        **kwargs
    ):
        super().__init__()

        # self.enc_p, self.dp, self.sdp all in one mxq
        self.enc_p_sdp_dp = MobilintTextEncoderAndDurationPredictor(
            n_vocab,
            hidden_channels,
            num_languages=num_languages,
            num_tones=num_tones,
            
            name_or_path=name_or_path,
            mxq_path=encoder_mxq_path,
            dev_no=dev_no,
            core_mode="single",
            target_cores=[target_core],
        )

        # self.dec, self.flow all in one mxq
        self.dec_flow = MobilintTransformerCouplingBlockAndGenerator(
            inter_channels,
            upsample_initial_channel,
            
            name_or_path=name_or_path,
            mxq_path=decoder_mxq_path,
            dev_no=dev_no,
            core_mode="single",
            target_cores=[target_core],
        )

        if n_speakers <= 0:
            logger.warning(
                "`self.n_speakers` should be positive to use g with self.emb_g."
            )

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
        length_scale=1.0,
        noise_scale_w=0.8,
        max_len=None,
        sdp_ratio=0.0,
        y=None,
        g=None,
    ):
        if x_lengths.shape != torch.Size([1]) or x_lengths[0] != int(x.shape[1]):
            logger.warning(
                f"Input `x_lengths` is set to `[x.shape[1]]` inside the mxq. x_length={x_lengths}, x_lengths[0]={x_lengths[0]}, x.shape={x.shape}"
            )

        if sid != 0:
            logger.warning("Input `sid` is set to 0 inside the mxq.")

        if g is not None:
            logger.warning(
                "Input `g` is calculated inside the mxq with assuming sid is 0."
            )

        if sdp_ratio != 0.2:
            logger.warning("Input `sdp_ratio` is set inside the mxq as 0.2.")

        if max_len is not None:
            logger.warning("Input `max_len` is not supported.")

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
        z, o = self.dec_flow(z_p)
        # print('max/min of o:', o.max(), o.min())
        return o, attn, y_mask, (z, z_p, m_p, logs_p)

    def launch(self):
        self.enc_p_sdp_dp.launch()
        self.dec_flow.launch()

    def dispose(self):
        self.enc_p_sdp_dp.dispose()
        self.dec_flow.dispose()
