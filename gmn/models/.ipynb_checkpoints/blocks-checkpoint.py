import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import WindowAttention, DropPath

class SWMSA(nn.Module):
    """
    Shifted-Windows Multi-Head Self-Attention (SW-MSA) block[cite: 218].
    We use the timm implementation of WindowAttention.
    The "shifting" is complex; for this implementation, we will use
    WindowAttention without the shift, which is a common simplification
    but captures the windowed self-attention idea.
    
    This block is designed to operate on feature maps (N, C, H, W).
    """
    def __init__(self, dim, num_heads, window_size=7):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = (window_size, window_size)
        
        self.attn = WindowAttention(
            dim=dim,
            num_heads=num_heads,
            window_size=self.window_size,
            qkv_bias=True,
            attn_drop=0.,
            proj_drop=0.
        )
    
    def forward(self, x):
        # Input x: [B, C, H, W]
        # Reshape to [B, H, W, C] for timm's attention
        x = x.permute(0, 2, 3, 1)
        
        B, H, W, C = x.shape
        
        # Pad features to be divisible by window size
        pad_b = (self.window_size[0] - H % self.window_size[0]) % self.window_size[0]
        pad_r = (self.window_size[1] - W % self.window_size[1]) % self.window_size[1]
        x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))
        
        _, H_pad, W_pad, _ = x.shape
        
        # Create windows
        x_windows = self.window_partition(x, self.window_size[0]) 
        # -> [B * num_windows, window_size, window_size, C]
        x_windows = x_windows.view(-1, self.window_size[0] * self.window_size[1], C) 
        # -> [B * num_windows, N, C]
        
        # Apply attention
        attn_windows = self.attn(x_windows) # [B * num_windows, N, C]
        
        # Reshape back
        attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
        x = self.window_reverse(attn_windows, self.window_size[0], H_pad, W_pad)
        
        # Remove padding
        x = x[:, :H, :W, :]
        
        # Reshape back to [B, C, H, W]
        x = x.permute(0, 3, 1, 2)
        
        return x

    def window_partition(self, x, window_size):
        B, H, W, C = x.shape
        x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
        return windows

    def window_reverse(self, windows, window_size, H, W):
        B = int(windows.shape[0] / (H * W / window_size / window_size))
        x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
        return x


class BD_Block(nn.Module):
    """
    Implements the Binocular Difference (BD) block from Fig 1(b) [cite: 104]
    and formulas (1)-(3)[cite: 225, 226].
    
    This implementation resolves the ambiguity in the paper by assuming all
    inputs and outputs are feature maps, to allow for hierarchical fusion.
    """
    def __init__(self, in_dim, prev_dim=None, num_heads=8, window_size=7, out_dim=256):
        super().__init__()
        
        # Left view path
        self.sw_msa = SWMSA(in_dim, num_heads, window_size)
        self.gelu1 = nn.GELU()
        
        # Right view path
        self.ln1 = nn.LayerNorm(in_dim)
        self.gelu2 = nn.GELU()
        
        # Hierarchical fusion
        self.downsampler = None
        current_concat_dim = in_dim
        
        if prev_dim is not None:
            # Add a downsampler for the previous hierarchical feature
            self.downsampler = nn.Sequential(
                nn.AvgPool2d(kernel_size=2, stride=2),
                nn.Conv2d(prev_dim, prev_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(prev_dim)
            )
            current_concat_dim += prev_dim

        # Final fusion layer (FC in paper[cite: 226], 1x1 Conv for maps)
        self.ln_out = nn.BatchNorm2d(current_concat_dim) # LN on 4D tensor is BN
        self.fc_out = nn.Conv2d(current_concat_dim, out_dim, kernel_size=1)

    def forward(self, f_l, f_r, bd_prev):
        # f_l, f_r: [B, C_in, H, W]
        
        # Left Path (Formula 1)
        f_l_attn = self.sw_msa(f_l)
        f_l_tilde = f_l + self.gelu1(f_l_attn) # [cite: 225]
        
        # Right Path (Formula 2)
        # LN on (B, C, H, W) needs (B, H, W, C)
        f_r_ln = self.ln1(f_r.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        f_r_tilde = f_r + self.gelu2(f_r_ln) # [cite: 225]
        
        # Difference
        diff = f_l_tilde - f_r_tilde # [cite: 226]
        
        # Hierarchical Concat (Formula 3)
        if bd_prev is not None:
            bd_prev_down = self.downsampler(bd_prev)
            x = torch.cat([bd_prev_down, diff], dim=1)
        else:
            x = diff
            
        # Output
        bd_out = self.fc_out(self.ln_out(x))
        return bd_out


class BS_Block(nn.Module):
    """
    Implements the Binocular Summation (BS) block from Fig 1(c)[cite: 119].
    
    This implementation resolves the Global Pooling contradiction [cite: 235]
    by omitting it, treating the FC as a 1x1 Conv, and keeping data as
    feature maps for hierarchical fusion, consistent with the FF block[cite: 278].
    """
    def __init__(self, in_dim, prev_dim=None, out_dim=256):
        super().__init__()
        
        # Summation path (FC in paper[cite: 119], 1x1 Conv for maps)
        self.fc = nn.Conv2d(in_dim, in_dim, kernel_size=1)
        
        # Hierarchical fusion
        self.downsampler = None
        current_concat_dim = in_dim
        
        if prev_dim is not None:
            # Add a downsampler for the previous hierarchical feature
            self.downsampler = nn.Sequential(
                nn.AvgPool2d(kernel_size=2, stride=2),
                nn.Conv2d(prev_dim, prev_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(prev_dim)
            )
            current_concat_dim += prev_dim
            
        # Output fusion layer
        self.out_conv = nn.Conv2d(current_concat_dim, out_dim, kernel_size=1)

    def forward(self, f_l, f_r, bs_prev):
        # f_l, f_r: [B, C_in, H, W]
        
        # Summation
        sum_feat = f_l + f_r
        fc_out = self.fc(sum_feat) # [cite: 136]
        
        # Hierarchical Concat (Formula 4) [cite: 237]
        if bs_prev is not None:
            bs_prev_down = self.downsampler(bs_prev)
            x = torch.cat([bs_prev_down, fc_out], dim=1)
        else:
            x = fc_out
        
        bs_out = self.out_conv(x)
        return bs_out


class FF_Block(nn.Module):
    """
    Implements the final Feature Fusion (FF) block from Fig 7 [cite: 278]
    and formulas (7)-(9)[cite: 283, 284, 287].
    """
    def __init__(self, bd_dim, mf_dim, bs_dim, num_heads=8, window_size=7):
        super().__init__()
        
        # BD Path (Formula 7)
        self.bd_sw_msa = SWMSA(bd_dim, num_heads, window_size)
        self.bd_gelu = nn.GELU()
        
        # BS Path (Formula 8)
        self.bs_ln = nn.LayerNorm(bs_dim)
        self.bs_gelu = nn.GELU()
        
        self.final_dim = bd_dim + mf_dim + bs_dim

    def forward(self, bd, mf, bs):
        # bd, mf, bs are all feature maps: [B, C, H, W]
        
        # BD Path
        bd_attn = self.bd_sw_msa(bd)
        bd_hat = bd + self.bd_gelu(bd_attn) # [cite: 283]
        
        # BS Path
        bs_ln = self.bs_ln(bs.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        bs_hat = bs + self.bs_gelu(bs_ln) # [cite: 284]
        
        # Final Concat (Formula 9)
        fused = torch.cat([bd_hat, mf, bs_hat], dim=1) # [cite: 287]
        
        return fused