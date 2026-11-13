import torch
import torch.nn as nn

class SkinConditionCNN(nn.Module):
    def __init__(self, num_classes: int = 35, in_channels: int = 3, dropout_rate: float = 0.4):
        super().__init__()

        #First 3 layers
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        #Next 3 layers
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        #Next 3 layers
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        #Next 3 layers
        self.block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Spatial adapter before classification: (256, 2, 2) -> 1024 elements
        # Note: (2, 2) ensures Apple Silicon MPS compatibility because spatial size 14 is divisible by 2.
        self.adaptive_pool = nn.AdaptiveAvgPool2d((2, 2))

        # Classifier head (Layers 13, 14, 15)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 2 * 2, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate * 0.75),
            nn.Linear(256, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        # Handle Apple Silicon MPS edge-case if non-divisible dimensions are ever passed
        if x.device.type == "mps" and (x.shape[2] % 2 != 0 or x.shape[3] % 2 != 0):
            x = self.adaptive_pool(x.to("cpu")).to(x.device)
        else:
            x = self.adaptive_pool(x)

        logits = self.classifier(x)
        return logits


if __name__ == "__main__":
    model = SkinConditionCNN(num_classes=35)
    dummy_input = torch.randn(2, 3, 224, 224)
    out = model(dummy_input)
    print("Forward pass successful!")
    print(f"Input shape:  {dummy_input.shape}")
    print(f"Output shape: {out.shape} (Expected: [2, 35])")
