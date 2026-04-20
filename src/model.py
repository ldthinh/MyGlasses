import torch
import torch.nn as nn
import torchvision.models as models


class FaceShapeModel(nn.Module):
    def __init__(self, num_classes=5, model_name='efficientnet_v2_s', pretrained=True, dropout=0.3):
        super(FaceShapeModel, self).__init__()

        if model_name == 'efficientnet_v2_s':
            weights = models.EfficientNet_V2_S_Weights.DEFAULT if pretrained else None
            self.backbone = models.efficientnet_v2_s(weights=weights)
            # Freeze toàn bộ backbone — Phase 1 training
            for param in self.backbone.parameters():
                param.requires_grad = False
            # Thay classifier head — chỉ phần này được train ở Phase 1
            in_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(in_features, num_classes)
            )

        elif model_name == 'efficientnet_b0':
            weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            self.backbone = models.efficientnet_b0(weights=weights)
            for param in self.backbone.parameters():
                param.requires_grad = False
            in_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(in_features, num_classes)
            )

        elif model_name == 'mobilenet_v3_small':
            weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
            self.backbone = models.mobilenet_v3_small(weights=weights)
            for param in self.backbone.parameters():
                param.requires_grad = False
            in_features = self.backbone.classifier[3].in_features
            self.backbone.classifier[3] = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(in_features, num_classes)
            )

        else:
            raise ValueError(f"Model '{model_name}' not supported. "
                             f"Choose from: efficientnet_v2_s, efficientnet_b0, mobilenet_v3_small")

        self.model_name = model_name

    def unfreeze_top_layers(self, num_blocks: int = 2):
        """
        Phase 2: Unfreeze `num_blocks` cuối của backbone để fine-tune.
        Chỉ mở một phần nhỏ, tránh catastrophic forgetting.
        """
        if self.model_name in ('efficientnet_v2_s', 'efficientnet_b0'):
            blocks = list(self.backbone.features.children())
            for block in blocks[-num_blocks:]:
                for param in block.parameters():
                    param.requires_grad = True

        elif self.model_name == 'mobilenet_v3_small':
            layers = list(self.backbone.features.children())
            for layer in layers[-num_blocks:]:
                for param in layer.parameters():
                    param.requires_grad = True

        # Luôn đảm bảo classifier được train
        for param in self.backbone.classifier.parameters():
            param.requires_grad = True

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"  Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

    def forward(self, x):
        return self.backbone(x)


if __name__ == "__main__":
    model = FaceShapeModel(num_classes=5, model_name='efficientnet_v2_s')
    print("=== Phase 1 (head only) ===")
    model.unfreeze_top_layers(0)

    print("\n=== Phase 2 (unfreeze 2 top blocks) ===")
    model.unfreeze_top_layers(2)

    x = torch.randn(1, 3, 150, 150)
    out = model(x)
    print(f"\nOutput shape: {out.shape}")
