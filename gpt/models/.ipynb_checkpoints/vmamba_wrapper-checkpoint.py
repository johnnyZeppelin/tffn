
import torch.nn as nn
import torch, os
# attempt to import the user's VMamba implementation if present in vmamba/ folder
def load_vmamba(root_dir):
    vmamba_path = os.path.join(root_dir, 'vmamba')
    if os.path.exists(vmamba_path):
        try:
            import sys
            sys.path.insert(0, vmamba_path)
            import vmamba  # expects vmamba package
            # assume vmamba.VMamba exists and returns a nn.Module
            if hasattr(vmamba, 'VMamba'):
                return vmamba.VMamba()
        except Exception as e:
            print('Failed to import VMamba from vmamba/:', e)
    # fallback small CNN
    class SmallCNN(nn.Module):
        def __init__(self, out_dim=512):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(3,32,3,padding=1), nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32,64,3,padding=1), nn.ReLU(),
                nn.AdaptiveAvgPool2d((1,1)),
            )
            self.fc = nn.Linear(64, out_dim)
        def forward(self, x):
            b = x.shape[0]
            y = self.net(x).view(b,-1)
            return self.fc(y)
    return SmallCNN()
