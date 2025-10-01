import torch
from torch import nn

from melo import commons
from melo.models import PosteriorEncoder, ReferenceEncoder

class SynthesizerTrn(nn.module):
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
        phone_tone_lang_emb_0 = (
            self.enc_p.emb(x)
            + self.enc_p.tone_emb(tone)
            + self.enc_p.language_emb(language)
        )
        z = (
            torch.randn(phone_tone_lang_emb_0.size(0), 2, phone_tone_lang_emb_0.size(1)).to(
                device=bert.device, dtype=bert.dtype
            )
            * noise_scale
        )

        z0 = z.unsqueeze(1)

        m_p, logs_p, logw = npu_model_0(
            z0[:, :, 1:2, :].permute(0, 1, 3, 2),
            phone_tone_lang_emb_0.unsqueeze(1),
            ja_bert.unsqueeze(1).permute(0, 1, 3, 2),
            z0[:, :, 0:1, :].flip((2,)).permute(0, 1, 3, 2),
        )

        x_mask = torch.ones_like(logw)
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
        
        z_p = z_p.unsqueeze(1).permute(0, 1, 3, 2) # (1, 1, W, C)
        npu_input = np.ascontiguousarray(z_p.to("cpu").numpy())
        npu_out = npu_model_1(npu_input)  # (1, seq_len, 1)
        o = torch.from_numpy(npu_out).squeeze(2).unsqueeze(0)  # (1, 1, seq_len)
        # print('max/min of o:', o.max(), o.min())
        return o, attn, y_mask, (None, z_p, m_p, logs_p)
