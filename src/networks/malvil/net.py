"""MaLViL: Multi-axis Low-rank Vision-LSTM for medical image segmentation.

Paper (MLMI @ MICCAI 2026): https://arxiv.org/abs/2608.17635

Modules (aligned with the paper):
  Bi-LRViL  — bidirectional ViL on a rank-p orthonormal subspace + learnable ω
  SaLViL    — scale-aware axis-aligned ConvX branches before Bi-LRViL
  CDM       — orthogonal H/V SaLViL paths with MAD fusion
  SGSM      — frequency-split skip modulation (g on smooth, β on detail)
  SQ-FFN    — spatial quadratic feed-forward (here: FFN)
"""

from functools import partial
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from timm.layers.drop import DropPath

from .shared import ResBlk, Upsample, Upsample4x, pvt_v2_b2
from .vision_lstm import SequenceTraversal as ST
from .vision_lstm import ViLBlock

# Paper default SaLViL configs per decoder stage (no L/G hints — ω is learnable).
# Format: p{rank}_k{kernels}_d{dilations}_s{channel_split_ratios}
# Hardcoded ranks/kernels: p32/k13 at 1/32, p64/k1357 … p256/k1357 at finer stages.
DEFAULT_STAGE_CFG: Dict[str, str] = {
    "4": "p32_k13_d11_s11",
    "3": "p64_k1357_d1111_s5551",
    "2": "p128_k1357_d1111_s5551",
    "1": "p256_k1357_d1111_s5551",
}


def parse_salvil_config(cfg: str):
    """Parse ``p{r}_k..._d..._s...`` (optional legacy ``tLG...`` token is ignored)."""
    rank = kernels = dilations = splits = None
    for part in cfg.split("_"):
        if not part:
            continue
        key, body = part[0], part[1:]
        if key == "p" and body.isdigit():
            rank = int(body)
        elif key == "k":
            kernels = [int(c) for c in body]
        elif key == "d":
            dilations = [int(c) for c in body]
        elif key == "s":
            splits = [int(c) for c in body]
        elif key == "t":
            continue  # legacy Local/Global init hints; ω is learnable
        else:
            raise ValueError(f"Unrecognized SaLViL config token '{part}' in '{cfg}'")

    if None in (rank, kernels, dilations, splits):
        raise ValueError(
            f"Incomplete SaLViL config '{cfg}'. Expected p{{r}}_k..._d..._s..."
        )
    if not (len(kernels) == len(dilations) == len(splits)):
        raise ValueError(
            f"Length mismatch in '{cfg}': kernels={kernels}, "
            f"dilations={dilations}, splits={splits}"
        )
    return rank, kernels, dilations, splits


# ---------------------------------------------------------------------------
# ConvX — axis-aligned (optionally sequential) residual convolutions
# ---------------------------------------------------------------------------
class ConvX(nn.Module):
    def __init__(self, in_chs: int, out_chs: int, ks: int, dilation: int, mode: str = "1d-full"):
        super().__init__()
        assert mode in ("1d-along", "1d-opposite", "1d-full", "2d-full")
        conv = partial(
            nn.Conv2d,
            in_channels=in_chs,
            out_channels=out_chs,
            stride=1,
            dilation=dilation,
            bias=False,
        )
        pad = ((ks - 1) // 2) * dilation
        if mode == "1d-along":
            core = conv(kernel_size=(1, ks), padding=(0, pad))
        elif mode == "1d-opposite":
            core = conv(kernel_size=(ks, 1), padding=(pad, 0))
        elif mode == "2d-full":
            core = conv(kernel_size=(ks, ks), padding=(pad, pad))
        else:  # 1d-full: sequential (1,k)+(k,1) ConvX (paper default; not a single k×1)
            core = nn.Sequential(
                conv(kernel_size=(1, ks), padding=(0, pad)),
                conv(kernel_size=(ks, 1), padding=(pad, 0)),
            )
        self.conv = nn.Sequential(core, nn.SiLU(inplace=True))

    def forward(self, x: Tensor) -> Tensor:
        return x + self.conv(x)


# ---------------------------------------------------------------------------
# SQ-FFN (Spatial Quadratic FFN)
# ---------------------------------------------------------------------------
class FFN(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        drop: float = 0.0,
    ):
        super().__init__()
        hidden_features = hidden_features or int(in_features * 3)
        out_features = out_features or in_features
        self.drop = nn.Dropout(drop)
        self.pwc1 = nn.Conv2d(in_features, hidden_features, 1, bias=False)
        self.dwc = nn.Conv2d(
            hidden_features, hidden_features, 3, 1, 1, groups=hidden_features, bias=False
        )
        self.s = nn.Parameter(torch.ones(1, hidden_features, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, hidden_features, 1, 1))
        self.pwc2 = nn.Conv2d(hidden_features, out_features, 1, bias=False)
        nn.init.zeros_(self.pwc2.weight)

    def forward(self, x: Tensor) -> Tensor:
        b, n, c = x.shape
        h = w = int(n**0.5)
        x = x.reshape(b, h, w, c).permute(0, 3, 1, 2)
        x = self.pwc1(x)
        x = self.dwc(x)
        x = F.relu(x)
        x = (x**2) * self.s + self.b
        x = self.drop(x)
        x = self.pwc2(x)
        x = self.drop(x)
        return x.permute(0, 2, 3, 1).reshape(b, n, -1)


# ---------------------------------------------------------------------------
# Bi-LRViL
# ---------------------------------------------------------------------------
class BiLowRankViL(nn.Module):
    r"""Bi-LRViL (paper eq. bilrvil).

    QR-orthonormal \(V\); \(Z=V^\top X\), \(\hat X = VZ\), \(X_\perp=X-\hat X\);
    opposite ViL scans mixed by \(\alpha\); residual gated by \(\omega\); MSE rec.
    """

    def __init__(self, dim: int, n_tokens: int, rank: int, stage: Optional[int] = None):
        super().__init__()
        self.dim = dim
        self.n_tokens = n_tokens
        self.stage = stage
        self.rank = min(rank, n_tokens)

        v = torch.randn(n_tokens, self.rank)
        v, _ = torch.linalg.qr(v)
        self.proj_down = nn.Parameter(v)

        self.bi_alpha = nn.Parameter(torch.zeros(dim))
        self.vil_fwd = ViLBlock(dim=dim, direction=ST.ROWWISE_FROM_TOP_LEFT)
        self.vil_bwd = ViLBlock(dim=dim, direction=ST.ROWWISE_FROM_BOT_RIGHT)
        # ω = σ(lg_gate); neutral init → ω≈0.5 (learned Local↔Global trade-off)
        self.lg_gate = nn.Parameter(torch.zeros(dim))
        self.register_buffer("gamma_scale", torch.ones(dim), persistent=False)
        self.last_rec_error = torch.tensor(0.0)

    def _basis(self) -> Tensor:
        q, _ = torch.linalg.qr(self.proj_down)
        return q

    def forward(self, x: Tensor) -> Tensor:
        b, n, c = x.shape
        assert n == self.n_tokens, f"Expected N={self.n_tokens}, got {n}"

        v = self._basis()  # QR orthonormal V
        u = v.t()

        x_comp = torch.einsum("bnc,nr->brc", x, v)  # Z = V^T X
        x_rec = torch.einsum("brc,rn->bnc", x_comp, u)
        self.last_rec_error = F.mse_loss(x, x_rec, reduction="mean")  # rec. term in eq. (objective)
        x_high = x - x_rec  # X_perp

        x_fwd = self.vil_fwd(x_comp)
        x_bwd = self.vil_bwd(x_comp)
        alpha = torch.sigmoid(self.bi_alpha).view(1, 1, c)
        x_vil = alpha * x_fwd + (1.0 - alpha) * x_bwd  # opposite ViL scans, α mix

        delta_comp = x_vil - x_comp
        delta_global = torch.einsum("brc,rn->bnc", delta_comp, u)
        omega = torch.sigmoid(self.lg_gate).view(1, 1, c)
        delta = delta_global - omega * x_high  # ω on residual

        return x + self.gamma_scale.view(1, 1, c) * delta


# ---------------------------------------------------------------------------
# SaLViL
# ---------------------------------------------------------------------------
class SaLViL(nn.Module):
    def __init__(
        self,
        dim: int,
        n_tokens: int,
        rank: int,
        stage: int,
        kernels: List[int],
        dilation_rates: List[int],
        channel_split: List[int],
        conv_mode: str = "1d-full",
    ):
        super().__init__()
        assert dim % sum(channel_split) == 0, (
            f"dim ({dim}) must be divisible by sum(channel_split)={sum(channel_split)}"
        )
        coeff = dim // sum(channel_split)
        self.channel_split = [c * coeff for c in channel_split]
        self.kernels = kernels

        self.cnas = nn.ModuleList()
        self.lorvils = nn.ModuleList()
        for split, ks, dr in zip(self.channel_split, kernels, dilation_rates):
            self.cnas.append(
                ConvX(split, split, ks=ks, dilation=dr, mode=conv_mode)
                if ks > 1
                else nn.Identity()
            )
            self.lorvils.append(BiLowRankViL(dim=split, n_tokens=n_tokens, rank=rank, stage=stage))

        self.mixer = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim, bias=False))
        nn.init.zeros_(self.mixer[1].weight)
        self.last_rec_error = torch.tensor(0.0)

    def forward(self, x_bchw: Tensor) -> Tensor:
        b, _, h, w = x_bchw.shape
        splits = torch.split(x_bchw, self.channel_split, dim=1)
        outs = []
        rec = x_bchw.new_zeros(())
        for i, xb in enumerate(splits):
            xb = self.cnas[i](xb)
            xb = xb.permute(0, 2, 3, 1).reshape(b, h * w, -1)
            xb = self.lorvils[i](xb)
            outs.append(xb)
            rec = rec + self.lorvils[i].last_rec_error
        x = torch.cat(outs, dim=-1)
        x = x + self.mixer(x)
        self.last_rec_error = rec / max(len(self.channel_split), 1)
        return x


# ---------------------------------------------------------------------------
# CDM — Cross-Directional Mixer
# ---------------------------------------------------------------------------
class CDM(nn.Module):
    r"""Cross-Directional Mixer (paper eq. cdm).

    Depthwise 3×3 stem, then H/V SaLViL (`rot90`), fused as \(\mu + \gamma_h A\).
    The 3×3 is applied before fusion, not as a parallel residual after.
    """

    def __init__(
        self,
        dim: int,
        stage: int,
        res: List[int],
        stage_cfg: Dict[str, str],
        num_rotations: int = 2,
        share_salvil_weights: bool = True,
        salvil_conv_mode: str = "1d-full",
    ):
        super().__init__()
        self.stage = stage
        self.num_rotations = num_rotations
        self.res = res
        self.share_salvil_weights = share_salvil_weights

        # Depthwise 3×3 stem (eq. cdm), then H/V SaLViL
        self.conv3 = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, groups=dim, bias=False),
            nn.BatchNorm2d(dim),
            nn.Conv2d(dim, dim, 1, bias=True),
        )
        self.act = nn.SiLU(inplace=True)

        n_tokens = res[0] * res[1]
        rank, kernels, dilations, splits = parse_salvil_config(stage_cfg[str(stage)])
        self.kernels = kernels

        make_salvil = partial(
            SaLViL,
            dim=dim,
            n_tokens=n_tokens,
            rank=rank,
            stage=stage,
            kernels=kernels,
            dilation_rates=dilations,
            channel_split=splits,
            conv_mode=salvil_conv_mode,
        )
        if share_salvil_weights:
            self.salvil_shared = make_salvil()
            self.salvils = None
        else:
            self.salvils = nn.ModuleList([make_salvil() for _ in range(num_rotations)])
            self.salvil_shared = None

        if num_rotations > 1:
            self.hf_gamma = nn.Parameter(torch.ones(dim, 1, 1) * 0.2)

        self.last_rec_error = torch.tensor(0.0)

    def forward(self, x: Tensor) -> Tensor:
        b, _, c = x.shape
        h, w = self.res
        x = x.reshape(b, h, w, c).permute(0, 3, 1, 2)
        x = self.act(self.conv3(x)) + x

        feats = []
        rec = torch.tensor(0.0, device=x.device)
        for i in range(self.num_rotations):
            x_rot = torch.rot90(x, k=i, dims=[2, 3]) if i > 0 else x
            salvil = self.salvil_shared if self.share_salvil_weights else self.salvils[i]
            y = salvil(x_rot)
            _, _, rh, rw = x_rot.shape
            y = y.reshape(b, rh, rw, c)
            if i > 0:
                y = torch.rot90(y, k=(4 - i), dims=[1, 2])
            feats.append(y.permute(0, 3, 1, 2))
            rec = rec + salvil.last_rec_error

        if self.num_rotations > 1:
            stack = torch.stack(feats, dim=0)
            mean = stack.mean(dim=0)
            # eq. (cdm): A = mean |F_i - μ|; fused = μ + γ_h A
            dev = (stack - mean.unsqueeze(0)).abs().mean(dim=0)
            fused = mean + self.hf_gamma.unsqueeze(0) * dev
            out = fused.flatten(2).transpose(1, 2)
        else:
            out = feats[0].flatten(2).transpose(1, 2)

        self.last_rec_error = rec / max(self.num_rotations, 1)
        return out


# ---------------------------------------------------------------------------
# MaLViL Block
# ---------------------------------------------------------------------------
class MaLViLBlock(nn.Module):
    r"""Decoder block: residual CDM then SQ-FFN, each with LayerScale \(\gamma\)."""

    def __init__(
        self,
        dim: int,
        stage: int,
        res: List[int],
        stage_cfg: Dict[str, str],
        norm_eps: float = 1e-5,
        hidden_ratio: float = 4.0,
        drop_prob: float = 0.0,
        num_rotations: int = 2,
        share_salvil_weights: bool = True,
        w_init: float = 1e-3,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=norm_eps)
        self.attn = CDM(
            dim=dim,
            stage=stage,
            res=res,
            stage_cfg=stage_cfg,
            num_rotations=num_rotations,
            share_salvil_weights=share_salvil_weights,
        )
        self.norm2 = nn.LayerNorm(dim, eps=norm_eps)
        stage_ratio = min(max(float(stage) / 4.0, 0.0), 1.0)
        self.mlp = FFN(
            in_features=dim,
            hidden_features=int(hidden_ratio * dim),
            out_features=dim,
            drop=0.5 * drop_prob * stage_ratio,
        )
        self.drop_path = DropPath(drop_prob=drop_prob)
        self.gamma_attn = nn.Parameter(torch.ones(dim) * w_init)
        self.gamma_mlp = nn.Parameter(torch.ones(dim) * w_init)
        self.last_rec_error = torch.tensor(0.0)

    def forward(self, x: Tensor) -> Tensor:
        b, c, h, w = x.shape
        x = x.view(b, c, h * w).transpose(-2, -1)
        # residual CDM then SQ-FFN, LayerScale γ
        x = x + self.drop_path(self.gamma_attn * self.attn(self.norm1(x)))
        x = x + self.drop_path(self.gamma_mlp * self.mlp(self.norm2(x)))
        self.last_rec_error = self.attn.last_rec_error
        return x.reshape(b, h, w, c).permute(0, 3, 1, 2)


# ---------------------------------------------------------------------------
# SGSM — Statistics-Guided Skip Modulation
# ---------------------------------------------------------------------------
class SGSM(nn.Module):
    r"""Statistics-Guided Skip Modulation (paper eq. sgsm).

    \(3\times3\) AvgPool split; \(q=[\mu,\mathrm{var},\mu(D),\max(D)]\);
    \(g=2\sigma(\mathrm{MLP}(q))\); \(D + g\odot S_l + \beta\odot S_h\).
    """

    def __init__(self, ch: int, reduction: int = 4):
        super().__init__()
        mid = max(ch // reduction, 8)
        self.fc = nn.Sequential(
            nn.Linear(ch * 4, mid, bias=False),
            nn.SiLU(inplace=True),
            nn.Linear(mid, ch, bias=False),
        )
        nn.init.zeros_(self.fc[-1].weight)
        self.detail_beta = nn.Parameter(torch.ones(ch))
        self.blur = nn.AvgPool2d(3, stride=1, padding=1)

    def forward(self, dec: Tensor, skip: Tensor) -> Tensor:
        s_low = self.blur(skip)
        s_high = skip - s_low
        stats = torch.cat(
            [
                s_low.mean(dim=[-2, -1]),
                s_low.var(dim=[-2, -1], unbiased=False),
                dec.mean(dim=[-2, -1]),
                dec.amax(dim=[-2, -1]),
            ],
            dim=1,
        )
        gate = (2.0 * torch.sigmoid(self.fc(stats))).view(dec.shape[0], -1, 1, 1)
        beta = self.detail_beta.view(1, -1, 1, 1)
        return dec + s_low * gate + s_high * beta


# ---------------------------------------------------------------------------
# Full network
# ---------------------------------------------------------------------------
class MaLViLNet(nn.Module):
    r"""Paper decoder: \(D_4=\mathcal{M}_4(E_4)\), \(D_i=\mathcal{M}_i(\mathrm{SGSM}(\mathcal{U}(D_{i+1}),E_i))\)."""

    def __init__(
        self,
        in_chs: int = 3,
        n_classes: int = 2,
        chs: Optional[List[int]] = None,
        img_res: Optional[List[int]] = None,
        norm_eps: float = 1e-5,
        hidden_ratio: float = 4.0,
        drop_prob: float = 0.05,
        encoder: str = "pvt_v2_b2",
        base_ptbbdir: Optional[str] = "",
        num_rotations: int = 2,
        share_salvil_weights: bool = True,
        decoder_stage_cfg: Optional[Dict[str, str]] = None,
        logger=None,
        **_unused,
    ) -> None:
        super().__init__()
        chs = chs or [64, 128, 320, 512]
        h, w = img_res or [224, 224]
        stage_cfg = decoder_stage_cfg or dict(DEFAULT_STAGE_CFG)
        log = logger.info if logger is not None else print
        self.num_rotations = num_rotations

        log(
            f"MaLViLNet: in={in_chs} classes={n_classes} chs={chs} "
            f"res=[{h},{w}] rotations={num_rotations} share_salvil={share_salvil_weights}"
        )
        log(f"  stage_cfg={stage_cfg}")

        if encoder != "pvt_v2_b2":
            raise ValueError(f"Release build supports encoder='pvt_v2_b2' only (got {encoder}).")

        self.backbone = pvt_v2_b2()
        if base_ptbbdir:
            ckpt = torch.load(f"{base_ptbbdir}/pvt/pvt_v2_b2.pth", map_location="cpu")
            model_dict = self.backbone.state_dict()
            model_dict.update({k: v for k, v in ckpt.items() if k in model_dict})
            self.backbone.load_state_dict(model_dict)
            print("Loaded pretrained PVTv2-B2 weights")

        make_dec = partial(
            MaLViLBlock,
            norm_eps=norm_eps,
            hidden_ratio=hidden_ratio,
            drop_prob=drop_prob,
            num_rotations=num_rotations,
            share_salvil_weights=share_salvil_weights,
            stage_cfg=stage_cfg,
        )
        make_up = partial(Upsample, kernel_size=3)

        self.de4 = make_dec(dim=chs[3], stage=4, res=[h // 32, w // 32], w_init=0.001)
        self.up4 = make_up(in_channels=chs[3], out_channels=chs[2])
        self.skde3 = SGSM(chs[2])
        self.de3 = make_dec(dim=chs[2], stage=3, res=[h // 16, w // 16], w_init=0.001)
        self.up3 = make_up(in_channels=chs[2], out_channels=chs[1])
        self.skde2 = SGSM(chs[1])
        self.de2 = make_dec(dim=chs[1], stage=2, res=[h // 8, w // 8], w_init=0.005)
        self.up2 = make_up(in_channels=chs[1], out_channels=chs[0])
        self.skde1 = SGSM(chs[0])
        self.de1 = make_dec(dim=chs[0], stage=1, res=[h // 4, w // 4], w_init=0.05)

        self.up1 = Upsample4x(in_channels=chs[0], out_channels=chs[0], kernel_size=7)
        self.resblk = ResBlk(in_chs=in_chs, out_chs=chs[0])
        self.head1 = nn.Conv2d(chs[0], n_classes, kernel_size=1, bias=True)
        self.last_rec_error = torch.tensor(0.0)

    def forward(self, x: Tensor):
        x_rb = self.resblk(x)
        x_in = x if x.shape[1] == 3 else x.repeat(1, 3, 1, 1)
        x1, x2, x3, x4 = self.backbone(x_in)

        # eq. (decoder): D_4 = M_4(E_4); D_i = M_i(SGSM(U(D_{i+1}), E_i))
        d4 = self.de4(x4)
        d3 = self.de3(self.skde3(self.up4(d4), x3))
        d2 = self.de2(self.skde2(self.up3(d3), x2))
        d1 = self.de1(self.skde1(self.up2(d2), x1))
        y = self.head1(self.up1(d1) + x_rb)

        # eq. (objective): rec averaged over stages (CDM already averages H/V SaLViL)
        self.last_rec_error = (
            self.de4.last_rec_error
            + self.de3.last_rec_error
            + self.de2.last_rec_error
            + self.de1.last_rec_error
        ) / 4.0
        return [y], self.last_rec_error
