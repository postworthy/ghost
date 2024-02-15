import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19

class MultiScalePerceptualColorLoss(nn.Module):
    def __init__(self, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        super().__init__()
        self.vgg = self._vgg_features().to(device)  # Move VGG to the specified device
        self.mse_loss = nn.MSELoss()
        self.device = device  # Store the device

    def _vgg_features(self):
        vgg_model = vgg19(pretrained=True).features
        for param in vgg_model.parameters():
            param.requires_grad = False
        return vgg_model.eval()

    def forward(self, Y, Xt):
        Y, Xt = Y.to(self.device), Xt.to(self.device)  # Ensure inputs are on the correct device
        # Compute multi-scale color consistency loss
        scales = [256, 128, 64, 32]  # Adjust based on your needs
        color_loss = 0.0
        for scale in scales:
            Y_down = F.interpolate(Y, size=(scale, scale), mode='bilinear', align_corners=False)
            Xt_down = F.interpolate(Xt, size=(scale, scale), mode='bilinear', align_corners=False)
            color_loss += self.mse_loss(Y_down, Xt_down)

        # Compute perceptual loss
        perceptual_loss = 0.0
        Y_features = self.vgg(Y)
        Xt_features = self.vgg(Xt)
        perceptual_loss = self.mse_loss(Y_features, Xt_features)

        # Combine losses
        total_loss = color_loss + perceptual_loss
        return total_loss
