import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
from diffusers.models.unet_2d_blocks import (
    CrossAttnDownBlock2D,
    CrossAttnUpBlock2D,
    DownBlock2D,
    UpBlock2D
)


def get_down_block(
    down_block_type,
    num_layers,
    in_channels,
    out_channels,
    temb_channels,
    add_downsample,
    resnet_eps,
    resnet_act_fn,
    attn_num_head_channels,
    resnet_groups=None,
    cross_attention_dim=None,
    downsample_padding=None,
    dual_cross_attention=False,
    use_linear_projection=False,
    only_cross_attention=False,
    upcast_attention=False,
    resnet_time_scale_shift="default",
    resnet_skip_time_act=False,
    resnet_out_scale_factor=1.0,
    cross_attention_norm=None,
):
    down_block_type = down_block_type[7:] if down_block_type.startswith("UNetRes") else down_block_type
    if down_block_type == "DownBlock2D":
        return DownBlock2D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            temb_channels=temb_channels,
            add_downsample=add_downsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            downsample_padding=downsample_padding,
            resnet_time_scale_shift=resnet_time_scale_shift,
        )
    elif down_block_type == "CrossAttnDownBlock2D":
        return VideoLDMDownBlock(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            temb_channels=temb_channels,
            add_downsample=add_downsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            downsample_padding=downsample_padding,
            cross_attention_dim=cross_attention_dim,
            attn_num_head_channels=attn_num_head_channels,
            dual_cross_attention=dual_cross_attention,
            use_linear_projection=use_linear_projection,
            only_cross_attention=only_cross_attention,
            upcast_attention=upcast_attention,
            resnet_time_scale_shift=resnet_time_scale_shift,
        )

    raise ValueError(f'{down_block_type} does not exist.')


def get_up_block(
    up_block_type,
    num_layers,
    in_channels,
    out_channels,
    prev_output_channel,
    temb_channels,
    add_upsample,
    resnet_eps,
    resnet_act_fn,
    attn_num_head_channels,
    resnet_groups=None,
    cross_attention_dim=None,
    dual_cross_attention=False,
    use_linear_projection=False,
    only_cross_attention=False,
    upcast_attention=False,
    resnet_time_scale_shift="default",
    resnet_skip_time_act=False,
    resnet_out_scale_factor=1.0,
    cross_attention_norm=None,
):
    up_block_type = up_block_type[7:] if up_block_type.startswith("UNetRes") else up_block_type
    if up_block_type == "UpBlock2D":
        return UpBlock2D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            prev_output_channel=prev_output_channel,
            temb_channels=temb_channels,
            add_upsample=add_upsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            resnet_time_scale_shift=resnet_time_scale_shift,
        )
    elif up_block_type == 'CrossAttnUpBlock2D':
        return VideoLDMUpBlock(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            prev_output_channel=prev_output_channel,
            temb_channels=temb_channels,
            add_upsample=add_upsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            cross_attention_dim=cross_attention_dim,
            attn_num_head_channels=attn_num_head_channels,
            dual_cross_attention=dual_cross_attention,
            use_linear_projection=use_linear_projection,
            only_cross_attention=only_cross_attention,
            upcast_attention=upcast_attention,
            resnet_time_scale_shift=resnet_time_scale_shift,
        )

    raise ValueError(f'{up_block_type} does not exist.')


class PositionalEncoding(nn.Module):
    def __init__(self, dim, max_pos=512):
        super().__init__()

        pos = torch.arange(max_pos)

        freq = torch.arange(dim//2) / dim
        freq = (freq * torch.tensor(10000).log()).exp()

        x = rearrange(pos, 'L -> L 1') / freq
        x = rearrange(x, 'L d -> L d 1')

        pe = torch.cat((x.sin(), x.cos()), dim=-1)
        self.pe = rearrange(pe, 'L d sc -> L (d sc)')

        self.dummy = nn.Parameter(torch.rand(1))

    def forward(self, length):
        enc = self.pe[:length]
        enc = enc.to(self.dummy.device)
        return enc


class Conv3DLayer(nn.Module):
    def __init__(self, in_dim, out_dim, n_frames):
        super().__init__()

        self.to_3d = Rearrange('(b t) c h w -> b c t h w', t=n_frames)
        self.to_2d = Rearrange('b c t h w -> (b t) c h w')

        k, p = (3, 1, 1), (1, 0, 0)
        self.block1 = nn.Sequential(
            nn.GroupNorm(32, in_dim),
            nn.SiLU(),
            nn.Conv3d(in_dim, out_dim, kernel_size=k, stride=1, padding=p)
        )
        self.block2 = nn.Sequential(
            nn.GroupNorm(32, out_dim),
            nn.SiLU(),
            nn.Conv3d(out_dim, out_dim, kernel_size=k, stride=1, padding=p)
        )

        self.alpha = nn.Parameter(torch.ones(1))

    def forward(self, x):
        h = self.to_3d(x)

        h = self.block1(h)
        h = self.block2(h)

        h = self.to_2d(h)

        with torch.no_grad():
            self.alpha.clamp_(0, 1)

        out = self.alpha * x + (1 - self.alpha) * h
        return out


class TemporalAttentionLayer(nn.Module):
    def __init__(self, dim, n_frames, n_heads=8):
        super().__init__()
        self.n_frames = n_frames

        self.pos_enc = PositionalEncoding(dim)

        head_dim = dim // n_heads
        proj_dim = head_dim * n_heads
        self.q_proj = nn.Linear(dim, proj_dim, bias=False)
        self.k_proj = nn.Linear(dim, proj_dim, bias=False)
        self.v_proj = nn.Linear(dim, proj_dim, bias=False)
        self.o_proj = nn.Linear(proj_dim, dim, bias=False)

        self.alpha = nn.Parameter(torch.ones(1))

    def forward(self, q, kv=None, mask=None):
        skip = q

        kv = kv if kv is not None else q

        bt, c, h, w = q.shape
        q = rearrange(q, '(b t) c h w -> (b h w) t c', t=self.n_frames)

        q = q + self.pos_enc(self.n_frames)

        qkv = torch.stack([self.q_proj(q), self.k_proj(kv), self.v_proj(kv)])
        q, k, v = rearrange(qkv, 'qkv bhw t (h d) -> qkv bhw h t d', h=self.n_heads)

        out = F.scaled_dot_product_attention(q, k, v, mask)
        out = rearrange(out, 'bhw h t d -> bhw t (h d)')
        out = self.o_proj(out)

        out = rearrange('(b h w) t c -> (b t) c h w', h=h, w=w)

        with torch.no_grad():
            self.alpha.clamp_(0, 1)

        out = self.alpha * skip + (1 - self.alpha) * out
        return out


class VideoLDMDownBlock(CrossAttnDownBlock2D):
    def __init__(self, *args, n_frames=8, n_heads=8, **kwargs):
        super().__init__(*args, **kwargs)

        in_channels = kwargs['in_channels']
        out_channels = kwargs['out_channels']
        num_layers = kwargs['num_layers']

        conv3ds = []
        tempo_attns = []

        for i in range(kwargs['num_layers']):
            in_channels = in_channels if i == 0 else out_channels

            conv3ds.append(
                Conv3DLayer(
                    in_dim=in_channels,
                    out_dim=out_channels,
                    n_frames=n_frames,
                )
            )

            tempo_attns.append(
                TemporalAttentionLayer(
                    dim=out_channels,
                    n_frames=n_frames,
                    n_heads=n_heads,
                )
            )

        self.conv3ds = nn.ModuleList(conv3ds)
        self.tempo_attns = nn.ModuleList(tempo_attns)

    def forward(
        self, hidden_states, temb=None, encoder_hidden_states=None, attention_mask=None, cross_attention_kwargs=None
    ):
        output_states = ()

        for resnet, conv3d, attn, tempo_attn in zip(self.resnets, self.conv3ds, self.attentions, self.tempo_attns):

            hidden_states = resnet(hidden_states, temb)
            hidden_states = conv3d(hidden_states)
            hidden_states = attn(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                cross_attention_kwargs=cross_attention_kwargs,
            ).sample
            hidden_states = tempo_attn(
                hidden_states,
                encoder_hidden_states,
            )

            output_states += (hidden_states,)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)

            output_states += (hidden_states,)

        return hidden_states, output_states


class VideoLDMUpBlock(CrossAttnUpBlock2D):
    def __init__(self, *args, n_frames=8, n_heads=8, **kwargs):
        super().__init__(*args, **kwargs)

        in_channels = kwargs['in_channels']
        out_channels = kwargs['out_channels']
        prev_output_channel = kwargs['prev_output_channel']
        num_layers = kwargs['num_layers']

        conv3ds = []
        tempo_attns = []

        for i in range(kwargs['num_layers']):
            res_skip_channels = in_channels if (i == num_layers - 1) else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels

            conv3ds.append(
                Conv3DLayer(
                    in_dim=resnet_in_channels + res_skip_channels,
                    out_dim=out_channels,
                    n_frames=n_frames,
                )
            )

            tempo_attns.append(
                TemporalAttentionLayer(
                    dim=out_channels,
                    n_frames=n_frames,
                    n_heads=n_heads,
                )
            )

        self.conv3ds = nn.ModuleList(conv3ds)
        self.tempo_attns = nn.ModuleList(tempo_attns)

    def forward(
        self,
        hidden_states,
        res_hidden_states_tuple,
        temb=None,
        encoder_hidden_states=None,
        cross_attention_kwargs=None,
        upsample_size=None,
        attention_mask=None,
    ):
        for resnet, conv3d, attn, tempo_attn in zip(self.resnets, self.conv3ds, self.attentions, self.tempo_attns):

            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

            hidden_states = resnet(hidden_states, temb)
            hidden_states = conv3d(hidden_states)
            hidden_states = attn(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                cross_attention_kwargs=cross_attention_kwargs,
            ).sample
            hidden_states = tempo_attn(
                hidden_states,
                encoder_hidden_states,
            )

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states, upsample_size)
        return hidden_states
