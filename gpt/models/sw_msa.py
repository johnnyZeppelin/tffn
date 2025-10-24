
import torch.nn as nn
import torch

class WindowedSelfAttention(nn.Module):
    \"\"\"A simplified windowed MHSA that splits spatial map into non-overlapping windows
       and applies nn.MultiheadAttention on flattened windows. This is an approximation
       of SW-MSA in the paper for reproducibility.
    \"\"\"
    def __init__(self, dim, num_heads=8, window_size=7):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
    def forward(self, x):
        # x: B x C x H x W
        B,C,H,W = x.shape
        ws = self.window_size
        # pad if necessary
        pad_h = (ws - (H % ws)) % ws
        pad_w = (ws - (W % ws)) % ws
        if pad_h or pad_w:
            x = nn.functional.pad(x, (0,pad_w,0,pad_h))
            _,_,H,W = x.shape
        # reshape into windows
        x = x.unfold(2, ws, ws).unfold(3, ws, ws) # B x C x nH x nW x ws x ws
        nH = x.size(2); nW = x.size(3)
        x = x.permute(0,2,3,1,4,5).contiguous() # B x nH x nW x C x ws x ws
        x = x.view(B*nH*nW, C, ws*ws).permute(0,2,1) # (B*nH*nW) x (ws*ws) x C
        # apply MHSA (treat tokens as sequence)
        out, _ = self.attn(x, x, x)  # (B*nH*nW) x (ws*ws) x C
        out = out.permute(0,2,1).contiguous().view(B, nH, nW, C, ws, ws)
        out = out.permute(0,3,1,4,2,5).contiguous().view(B, C, nH*ws, nW*ws)
        # crop to original H W if padded
        out = out[:,:,:H-pad_h if pad_h else H, :,:W-pad_w if pad_w else W]
        return out
