import torch
import torch.nn as nn
import torchvision.models as models


class HybridCNNTransformer(nn.Module):
    def __init__(
        self,
        num_classes: int = 3,
        pretrained: bool = True,
        strict_pretrained: bool = True,
    ):
        super(HybridCNNTransformer, self).__init__()

        # CNN backbone (ResNet18)
        if pretrained:
            try:
                try:
                    weights = models.ResNet18_Weights.DEFAULT
                    resnet = models.resnet18(weights=weights)
                except AttributeError:
                    # Older torchvision API
                    resnet = models.resnet18(weights="DEFAULT")
            except Exception as e:
                if strict_pretrained:
                    raise RuntimeError(
                        "Failed to load pretrained ResNet18 weights. "
                        "Please fix internet/SSL to download weights, then retry."
                    ) from e
                print(
                    f"Warning: could not load pretrained ResNet18 weights "
                    f"(reason: {e}). Falling back to random initialization."
                )
                try:
                    resnet = models.resnet18(weights=None)
                except TypeError:
                    # Very old API
                    resnet = models.resnet18(pretrained=False)
        else:
            try:
                resnet = models.resnet18(weights=None)
            except TypeError:
                resnet = models.resnet18(pretrained=False)

        # Remove the final pooling and FC layers
        self.cnn = nn.Sequential(*list(resnet.children())[:-2])

        # Project from 512 channels to 256 for the transformer
        self.conv = nn.Conv2d(512, 256, kernel_size=1)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=256,
            nhead=8,
            batch_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=2,
        )

        # Pool over sequence length
        self.pool = nn.AdaptiveAvgPool1d(1)

        # Classification head
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 3, 224, 224]
        x = self.cnn(x)  # [B, 512, H, W]
        x = self.conv(x)  # [B, 256, H, W]

        B, C, H, W = x.shape

        # Flatten spatial dims and treat as sequence
        x = x.view(B, C, H * W).permute(0, 2, 1)  # [B, HW, 256]

        x = self.transformer(x)  # [B, HW, 256]

        x = x.permute(0, 2, 1)  # [B, 256, HW]
        x = self.pool(x).squeeze(-1)  # [B, 256]

        out = self.fc(x)  # [B, num_classes]

        return out