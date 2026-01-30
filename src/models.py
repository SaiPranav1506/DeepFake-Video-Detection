import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    import timm
except Exception:
    timm = None

try:
    from transformers import CLIPVisionModel
except Exception:
    CLIPVisionModel = None

try:
    from transformers import Dinov2Model
except Exception:
    Dinov2Model = None


class CNNLSTMHybrid(nn.Module):
    def __init__(self, input_channels=3, hidden_size=256, num_layers=2, num_classes=2, dropout=0.3):
        super(CNNLSTMHybrid, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.cnn = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        
        self.cnn_out_features = 512
        
        self.lstm = nn.LSTM(
            input_size=self.cnn_out_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        cnn_features = self.cnn(x)
        cnn_features = cnn_features.view(B, T, self.cnn_out_features)
        
        lstm_out, (h_n, c_n) = self.lstm(cnn_features)
        
        attention_weights = self.attention(lstm_out)
        attention_weights = torch.softmax(attention_weights, dim=1)
        context = torch.sum(attention_weights * lstm_out, dim=1)
        
        out = self.classifier(context)
        return out


class ViTFeatureExtractor(nn.Module):
    def __init__(self, model_name='vit_base_patch16_224', pretrained=False, out_dim=768):
        super().__init__()
        self.out_dim = out_dim
        if timm is not None:
            self.vit = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
            self.out_dim = self.vit.num_features
        else:
            # fallback: small conv net (very small). Encourage installing timm for production.
            self.vit = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(64, out_dim)
            )

    def forward(self, x):
        # x: (B, 3, H, W)
        return self.vit(x)


class CLIPVisionFeatureExtractor(nn.Module):
    """CLIP vision encoder wrapper that returns a pooled image embedding.

    Uses HuggingFace Transformers' `CLIPVisionModel`.

    Input expects float tensors in [0, 1] range with shape (B, 3, H, W).
    """

    # CLIP image normalization constants
    _MEAN = (0.48145466, 0.4578275, 0.40821073)
    _STD = (0.26862954, 0.26130258, 0.27577711)

    def __init__(self, model_name: str = 'openai/clip-vit-base-patch32', pretrained: bool = True):
        super().__init__()
        if CLIPVisionModel is None:
            raise RuntimeError(
                "`transformers` is required for CLIPVisionFeatureExtractor. Install it (pip install transformers)."
            )

        if pretrained:
            self.clip = CLIPVisionModel.from_pretrained(model_name)
        else:
            # build from config inferred from pretrained weights (still requires download)
            self.clip = CLIPVisionModel.from_pretrained(model_name)
            self.clip.apply(self._init_weights)

        self.out_dim = int(getattr(self.clip.config, 'hidden_size', 768))

        mean = torch.tensor(self._MEAN, dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor(self._STD, dtype=torch.float32).view(1, 3, 1, 1)
        self.register_buffer('clip_mean', mean, persistent=False)
        self.register_buffer('clip_std', std, persistent=False)

    @staticmethod
    def _init_weights(m):
        # lightweight init for modules if user explicitly sets pretrained=False
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # x: (B, 3, H, W) float in [0, 1]
        x = (x - self.clip_mean) / self.clip_std
        out = self.clip(pixel_values=x)
        # Prefer pooler_output if present, otherwise CLS token
        pooled = getattr(out, 'pooler_output', None)
        if pooled is None:
            pooled = out.last_hidden_state[:, 0, :]
        return pooled


class DINOv2VisionFeatureExtractor(nn.Module):
    """DINOv2 vision encoder wrapper.

    Uses HuggingFace Transformers' `Dinov2Model`.

    Input expects float tensors in [0, 1] range with shape (B, 3, H, W).
    Applies ImageNet normalization by default.
    """

    _MEAN = (0.485, 0.456, 0.406)
    _STD = (0.229, 0.224, 0.225)

    def __init__(self, model_name: str = 'facebook/dinov2-base', pretrained: bool = True):
        super().__init__()
        if Dinov2Model is None:
            raise RuntimeError(
                "`transformers` is required for DINOv2VisionFeatureExtractor. Install it (pip install transformers)."
            )

        # Note: Dinov2Model.from_pretrained is the standard path.
        # `pretrained=False` isn't well-defined without a config, so we keep the same load path.
        self.dino = Dinov2Model.from_pretrained(model_name) if pretrained else Dinov2Model.from_pretrained(model_name)

        self.out_dim = int(getattr(self.dino.config, 'hidden_size', 768))

        mean = torch.tensor(self._MEAN, dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor(self._STD, dtype=torch.float32).view(1, 3, 1, 1)
        self.register_buffer('dino_mean', mean, persistent=False)
        self.register_buffer('dino_std', std, persistent=False)

    def forward(self, x):
        x = (x - self.dino_mean) / self.dino_std
        out = self.dino(pixel_values=x)
        # Use CLS token embedding
        return out.last_hidden_state[:, 0, :]


class SimpleGCN(nn.Module):
    """A small spectral GCN implemented without external GNN libs.

    Forward: H' = ReLU(A_norm @ H @ W)
    """
    def __init__(self, in_dim, hid_dim=256, out_dim=128, dropout=0.3):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hid_dim)
        self.fc2 = nn.Linear(hid_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, H, A_norm):
        # H: (B, N, F); A_norm: (B, N, N)
        # message passing: A_norm @ H -> (B, N, F)
        H = torch.bmm(A_norm, H)
        H = self.fc1(H)
        H = F.relu(H)
        H = self.dropout(H)
        H = self.fc2(H)
        H = F.relu(H)
        return H


class DeepfakeModel(nn.Module):
    def __init__(
        self,
        vit_out=768,
        gcn_hid=256,
        gcn_out=128,
        num_classes=2,
        pretrained_vit=False,
        vit_model_name='vit_base_patch16_224',
        vit_pretrained_path=None,
        backbone: str = 'timm_vit',
        clip_model_name: str = 'openai/clip-vit-base-patch32',
        clip_pretrained: bool = True,
        dinov2_model_name: str = 'facebook/dinov2-base',
        dinov2_pretrained: bool = True,
    ):
        super().__init__()

        # Vision backbone
        backbone = (backbone or 'timm_vit').lower()
        if backbone in {'clip', 'clip_vit', 'clip-vit', 'clipvit'}:
            self.vit = CLIPVisionFeatureExtractor(model_name=clip_model_name, pretrained=clip_pretrained)
        elif backbone in {'dinov2', 'dino_v2', 'dinov2_vit', 'dinov2-vit', 'dinov2vit'}:
            self.vit = DINOv2VisionFeatureExtractor(model_name=dinov2_model_name, pretrained=dinov2_pretrained)
        else:
            # allow using a timm pretrained ViT backbone
            self.vit = ViTFeatureExtractor(model_name=vit_model_name, pretrained=pretrained_vit, out_dim=vit_out)

        inferred_vit_out = int(getattr(self.vit, 'out_dim', vit_out))
        self.vit_proj = nn.Identity() if inferred_vit_out == int(vit_out) else nn.Linear(inferred_vit_out, int(vit_out))

        # optionally load user-supplied vit weights
        if vit_pretrained_path is not None:
            try:
                state = torch.load(vit_pretrained_path, map_location='cpu')
                # attempt to load into vit module
                if isinstance(state, dict) and ('model' in state or 'state_dict' in state or 'model_state' in state):
                    sd = state.get('model', state.get('state_dict', state.get('model_state', state)))
                else:
                    sd = state
                # try without prefix
                try:
                    self.vit.load_state_dict(sd)
                except Exception:
                    # try stripping common prefix
                    sd2 = {k.replace('base_model.encoder.', ''): v for k, v in sd.items()}
                    self.vit.load_state_dict(sd2, strict=False)
                print(f'Loaded ViT pretrained weights from {vit_pretrained_path}')
            except Exception as e:
                print(f'Warning: failed to load ViT weights from {vit_pretrained_path}: {e}')
        self.gcn = SimpleGCN(in_dim=vit_out, hid_dim=gcn_hid, out_dim=gcn_out)
        self.classifier = nn.Sequential(
            nn.Linear(gcn_out, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, images, A_norm):
        # images: (B, N, 3, H, W) -> flatten nodes into batch for ViT
        B, N, C, H, W = images.shape
        x = images.view(B * N, C, H, W)
        feats = self.vit(x)  # (B*N, F_backbone)
        feats = self.vit_proj(feats)  # (B*N, vit_out)
        feats = feats.view(B, N, -1)  # (B, N, F)
        g = self.gcn(feats, A_norm)  # (B, N, gcn_out)
        # global pooling (mean)
        g_pool = g.mean(dim=1)  # (B, gcn_out)
        out = self.classifier(g_pool)
        return out
