import math
import torch
from torch import nn

from melo import commons
from melo.models import PosteriorEncoder, ReferenceEncoder

import maccel
from maccel import Cluster, Core, CoreId
import numpy as np

class MobilintTextEncoderAndDurationPredictor(nn.Module):
    def __init__(
        self,
        mxq_path,
    ):
        super().__init__()
        
        self.acc = maccel.Accelerator()
        mc0 = maccel.ModelConfig()
        mc0.set_single_core_mode(core_ids=[CoreId(Cluster.Cluster0, Core.Core0)])
        self.model = maccel.Model(mxq_path, mc0)
        self.model.launch(self.acc)
    
    def text_encoder_forward(self, x, x_lengths, tone, language, bert, ja_bert, g=None):
        bert_emb = self.bert_proj(bert).transpose(1, 2)
        ja_bert_emb = self.ja_bert_proj(ja_bert).transpose(1, 2)
        x = (
            self.emb(x)
            + self.tone_emb(tone)
            + self.language_emb(language)
            + bert_emb
            + ja_bert_emb
        ) * math.sqrt(
            self.hidden_channels
        )  # [b, t, h]
        x = torch.transpose(x, 1, -1)  # [b, h, t]
        x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(
            x.dtype
        )

        x = self.encoder(x * x_mask, x_mask, g=g)
        stats = self.proj(x) * x_mask

        m, logs = torch.split(stats, self.out_channels, dim=1)
        return x, m, logs, x_mask

    def duraction_predictor_forward(self, x, x_mask, g=None):
        x = torch.detach(x)
        if g is not None:
            g = torch.detach(g)
            x = x + self.cond(g)
        x = self.conv_1(x * x_mask)
        x = torch.relu(x)
        x = self.norm_1(x)
        x = self.drop(x)
        x = self.conv_2(x * x_mask)
        x = torch.relu(x)
        x = self.norm_2(x)
        x = self.drop(x)
        x = self.proj(x * x_mask)
        return x * x_mask
    
    def __call__(
        self,
        x,
        tone,
        language,
        bert,
        noise_scale,
        axis=2
    ):
        phone_tone_lang_emb_0 = (
            self.emb(x)
            + self.tone_emb(tone)
            + self.language_emb(language)
        )
        
        z = (
            torch.randn(phone_tone_lang_emb_0.size(0), 2, phone_tone_lang_emb_0.size(1)).to(
                device=bert.device, dtype=bert.dtype
            )
            * noise_scale
        )
        
        z0 = z.unsqueeze(1)
        
        z_concat_flip_split = z0[:, :, 1:2, :].permute(0, 1, 3, 2).numpy()
        phone_tone_lang_emb_0 = phone_tone_lang_emb_0.unsqueeze(1).numpy()
        ja_bert = ja_bert.unsqueeze(1).permute(0, 1, 3, 2).numpy()
        z_flip_split = z0[:, :, 0:1, :].flip((2,)).permute(0, 1, 3, 2).numpy()
        
        allowed_chunks = [100, 200, 300, 400]  # largest-first
        max_chunk = max(allowed_chunks)
        
        _, _, origin_seq, _ = z_concat_flip_split.shape
        print(f"[DEBUG-TRN0] original_seq: {origin_seq}")
        
        m_p_chunks, logs_p_chunks, logs_w_chunks = [], [], []
        
        cur_seq_len = 0
        while cur_seq_len < origin_seq:
            print("cur_seq_len: ", cur_seq_len, " origin_seq: ", origin_seq)
            remaining = origin_seq - cur_seq_len
            
            # Pick largest allowed chunk <= remaining
            chunk_size = None
            for chunk in allowed_chunks:
                if chunk >= remaining:
                    chunk_size = chunk
                    break
            if chunk_size is None:
                # No chunk >= remaining → use max chunk
                chunk_size = max_chunk
            
            next_seq_len = cur_seq_len + min(chunk_size, remaining)
            # Slice inputs
            slices = [slice(None)] * z_concat_flip_split.ndim
            slices[axis] = slice(cur_seq_len, next_seq_len)
            
            input0_slice = z_concat_flip_split[tuple(slices)].astype(np.float32)
            input1_slice = phone_tone_lang_emb_0[tuple(slices)].astype(np.float32)
            input2_slice = ja_bert[tuple(slices)].astype(np.float32)
            input3_slice = z_flip_split[tuple(slices)].astype(np.float32)

            pad_len = chunk_size - (next_seq_len - cur_seq_len)
            if remaining < chunk_size:
                pad_width = [(0, 0)] * z_concat_flip_split.ndim
                pad_width[axis] = (0, pad_len)
                input0_slice = np.pad(input0_slice, pad_width, mode="constant", constant_values=0)
                input1_slice = np.pad(input1_slice, pad_width, mode="constant", constant_values=0)
                input2_slice = np.pad(input2_slice, pad_width, mode="constant", constant_values=0)
                input3_slice = np.pad(input3_slice, pad_width, mode="constant", constant_values=0)
            
            # Inference
            outputs = self.model.infer([input0_slice, input1_slice, input2_slice, input3_slice])
            
            m_p_chunks.append(outputs[0])
            logs_p_chunks.append(outputs[1])
            logs_w_chunks.append(outputs[2])
            
            cur_seq_len = next_seq_len
        
        # Concatenate along sequence axis
        m_p = np.concatenate(m_p_chunks, axis=axis)
        logs_p = np.concatenate(logs_p_chunks, axis=axis)
        logs_w = np.concatenate(logs_w_chunks, axis=axis)
        
        # Trim to original sequence length
        m_p = m_p[..., :origin_seq, :]
        logs_p = logs_p[..., :origin_seq, :]
        logs_w = logs_w[..., :origin_seq]
        
        # Convert to torch tensors
        m_p = torch.from_numpy(m_p).squeeze(1).transpose(1, 2)
        logs_p = torch.from_numpy(logs_p).squeeze(1).transpose(1, 2)
        logs_w = torch.from_numpy(logs_w)
        x_mask = torch.ones_like(logs_w)
        
        return m_p, logs_p, x_mask, logs_w

class MobilintCouplingBlockAndGenerator(nn.Module):
    def __init__(
        self,
        mxq_path,
        language,
    ):
        self.acc = maccel.Accelerator()
        mc1 = maccel.ModelConfig()
        mc1.set_single_core_mode(core_ids=[CoreId(Cluster.Cluster0, Core.Core2)])
        self.model = maccel.Model(mxq_path, mc1)
        self.model.launch(self.acc)
        input_shape_info = self.model.get_model_input_shape()
        _, self.seq_len, self.channels = input_shape_info[0]  # (1, W, C)
        if language == "KR":
            self.allowed_chunks = [200, 300, 400, 500, 600, 900]
        elif language == "EN":
            self.allowed_chunks = [300]
        else:
            raise ValueError(f"language: {language} is not supported")
        
    def __call__(self, z_p, channels=96, axis=2):
        z_p = z_p.unsqueeze(1).permute(0, 1, 3, 2) # (1, 1, W, C)
        z_p = np.ascontiguousarray(z_p.to("cpu").numpy()) # (1, seq_len, 1)
        
        allowed_chunks = self.allowed_chunks
        max_chunk = max(allowed_chunks)
        _, _, origin_seq, _ = z_p.shape
        # 1. Pad
        output_chunks = []
        cur_seq_len = 0
        while cur_seq_len < origin_seq:
            remaining = origin_seq - cur_seq_len
            
            # Pick largest allowed chunk <= remaining
            chunk_size = None
            for chunk in allowed_chunks:
                if chunk >= remaining:
                    chunk_size = chunk
                    break
            if chunk_size is None:
                # No chunk >= remaining → use max chunk
                chunk_size = max_chunk

            next_seq_len = cur_seq_len + min(chunk_size, remaining)
            slices = [slice(None)] * z_p.ndim
            slices[axis] = slice(cur_seq_len, next_seq_len)
            z_p_slice = z_p[tuple(slices)].astype(np.float32)
            pad_len = chunk_size - (next_seq_len - cur_seq_len)
            if remaining < chunk_size:
                pad_width = [(0, 0)] * z_p_slice.ndim
                pad_width[axis] = (0, pad_len)
                z_p_slice = np.pad(z_p_slice, pad_width, mode="constant", constant_values=0)

            x_0 = z_p_slice[0,..., channels :]
            x_1 = z_p_slice[0,..., channels - 1 :: -1]
            out = self.model.infer([x_0, x_1])
            output_chunks.append(out[0])

            cur_seq_len = next_seq_len

        new_len = origin_seq*512
        audio = np.concatenate(output_chunks, 1)[:,:new_len,:].transpose(0,2,1)
        audio = torch.from_numpy(audio).squeeze(2).unsqueeze(0)  # (1, 1, seq_len)

        return None, audio

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
        **kwargs
    ):
        super().__init__()
        self.n_vocab = n_vocab
        self.spec_channels = spec_channels
        self.inter_channels = inter_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.resblock = resblock
        self.resblock_kernel_sizes = resblock_kernel_sizes
        self.resblock_dilation_sizes = resblock_dilation_sizes
        self.upsample_rates = upsample_rates
        self.upsample_initial_channel = upsample_initial_channel
        self.upsample_kernel_sizes = upsample_kernel_sizes
        self.segment_size = segment_size
        self.n_speakers = n_speakers
        self.gin_channels = gin_channels
        self.n_layers_trans_flow = n_layers_trans_flow
        self.use_spk_conditioned_encoder = kwargs.get(
            "use_spk_conditioned_encoder", True
        )
        self.use_sdp = use_sdp
        self.use_noise_scaled_mas = kwargs.get("use_noise_scaled_mas", False)
        self.mas_noise_scale_initial = kwargs.get("mas_noise_scale_initial", 0.01)
        self.noise_scale_delta = kwargs.get("noise_scale_delta", 2e-6)
        self.current_mas_noise_scale = self.mas_noise_scale_initial
        if self.use_spk_conditioned_encoder and gin_channels > 0:
            self.enc_gin_channels = gin_channels
        else:
            self.enc_gin_channels = 0
        # self.enc_p, self.dp all in one mxq, self.sdp is not in because sdp_ratio = 0
        self.enc_p_sdp_dp = MobilintTextEncoderAndDurationPredictor(
        )
        # self.dec, self.flow all in one mxq
        self.dec_flow = MobilintCouplingBlockAndGenerator(
        )
        self.enc_q = PosteriorEncoder(
            spec_channels,
            inter_channels,
            hidden_channels,
            5,
            1,
            16,
            gin_channels=gin_channels,
        )

        if n_speakers > 0:
            self.emb_g = nn.Embedding(n_speakers, gin_channels)
        else:
            self.ref_enc = ReferenceEncoder(spec_channels, gin_channels, layernorm=norm_refenc)
        self.use_vc = use_vc

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
        # x, m_p, logs_p, x_mask = self.enc_p(x, x_lengths, tone, language, bert)
        # g = self.gst(y)
        if g is None:
            if self.n_speakers > 0:
                g = self.emb_g(sid).unsqueeze(-1)  # [b, h, 1]
            else:
                g = self.ref_enc(y.transpose(1, 2)).unsqueeze(-1)
        if self.use_vc:
            g_p = None
        else:
            g_p = g
        ###########################################
        x, m_p, logs_p, x_mask = self.enc_p(
            x, x_lengths, tone, language, bert, ja_bert, g=g_p
        )
        logw = self.sdp(x, x_mask, g=g, reverse=True, noise_scale=noise_scale_w) * (
            sdp_ratio
        ) + self.dp(x, x_mask, g=g) * (1 - sdp_ratio)
        ###########################################
        m_p, logs_p, x_mask, logw = self.enc_p_sdp_dp(
            x,
            tone,
            language,
            bert,
            noise_scale,
        )
        ###########################################
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
        ###########################################
        z = self.flow(z_p, y_mask, g=g, reverse=True)
        o = self.dec((z * y_mask)[:, :, :max_len], g=g)
        ###########################################
        z, o = self.flow_dec(z_p)
        ###########################################
        # print('max/min of o:', o.max(), o.min())
        return o, attn, y_mask, (z, z_p, m_p, logs_p)
