"""
Enhanced Siamese Retrieval Detector V2 with EfficientNet-B3 and Attention Mechanisms
- EfficientNet-B3 backbone
- Patch-based search image processing (8 patches: 2x4 grid)
- Self-attention between patches
- Cross-attention with reference features
- Fixed input size: 640x640, embed_dim: 512
"""

import math
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


def build_efficientnet_backbone(pretrained: bool = True, out_channels: int = 512) -> nn.Module:
    """
    Build EfficientNet-B3 backbone with custom output projection.
    
    Args:
        pretrained: Use pretrained weights
        out_channels: Output feature channels (512)
        
    Returns:
        backbone: Feature extractor with projection head
    """
    # Load EfficientNet-B3
    backbone = models.efficientnet_b3(pretrained=pretrained)
    
    # Remove classifier head, keep only feature extractor
    feature_extractor = backbone.features
    
    # Auto-detect actual output channels with dummy input
    with torch.no_grad():
        dummy_input = torch.randn(1, 3, 224, 224)
        dummy_output = feature_extractor(dummy_input)
        actual_in_channels = dummy_output.shape[1]
    
    # Projection head: actual_in_channels -> out_channels
    projection = nn.Sequential(
        nn.Conv2d(actual_in_channels, out_channels, kernel_size=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )
    
    return nn.Sequential(feature_extractor, projection)


class PatchEmbedding(nn.Module):
    """
    Split image features into 16 patches (4x4 grid) and create embeddings.
    """
    def __init__(self, embed_dim: int = 512):
        super().__init__()
        # Updated configuration for better small object detection
        self.grid_h, self.grid_w = 4, 4  # 4x4 = 16 patches
        self.num_patches = 16
        self.embed_dim = embed_dim
        
        # 2D Positional embeddings for patches (row, col encoding)
        self.pos_embed_h = nn.Parameter(torch.randn(1, self.grid_h, 1, embed_dim // 2) * 0.02)
        self.pos_embed_w = nn.Parameter(torch.randn(1, 1, self.grid_w, embed_dim // 2) * 0.02)
        
        # Adaptive projection layer (will be created dynamically)
        self.adaptive_proj = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, embed_dim, H_feat, W_feat) - Features from backbone
            
        Returns:
            patches: (B, num_patches, embed_dim) - Patch embeddings
        """
        B, C, H, W = x.shape
        
        # Divide feature map into patches
        patch_h = H // self.grid_h
        patch_w = W // self.grid_w
        
        # Reshape to patches: (B, C, grid_h, patch_h, grid_w, patch_w)
        x = x.view(B, C, self.grid_h, patch_h, self.grid_w, patch_w)
        
        # Rearrange to: (B, grid_h, grid_w, C, patch_h, patch_w)
        x = x.permute(0, 2, 4, 1, 3, 5)
        
        # Flatten patches: (B, num_patches, C * patch_h * patch_w)
        patch_dim = C * patch_h * patch_w
        x = x.contiguous().view(B, self.num_patches, patch_dim)
        
        # Project to embed_dim if needed
        if patch_dim != self.embed_dim:
            if self.adaptive_proj is None:
                self.adaptive_proj = nn.Linear(patch_dim, self.embed_dim).to(x.device)
            x = self.adaptive_proj(x)
        
        # Add 2D positional embeddings
        # Reshape to (B, grid_h, grid_w, embed_dim)
        x_2d = x.view(B, self.grid_h, self.grid_w, self.embed_dim)
        
        # Split embedding dimension for row and column encoding
        x_2d_split = x_2d.view(B, self.grid_h, self.grid_w, 2, self.embed_dim // 2)
        
        # Add positional embeddings
        pos_h = self.pos_embed_h.expand(B, self.grid_h, self.grid_w, self.embed_dim // 2)
        pos_w = self.pos_embed_w.expand(B, self.grid_h, self.grid_w, self.embed_dim // 2)
        
        x_2d_split[:, :, :, 0] += pos_h
        x_2d_split[:, :, :, 1] += pos_w
        
        # Reshape back to (B, num_patches, embed_dim)
        x = x_2d_split.view(B, self.num_patches, self.embed_dim)
        
        return x


class MultiHeadAttention(nn.Module):
    """Multi-head attention module."""
    
    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert embed_dim % num_heads == 0
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, N, C = x.shape
        
        # Generate Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        
        return x


class CrossAttention(nn.Module):
    """Cross-attention between search patches (Q) and reference features (K, V)."""
    
    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert embed_dim % num_heads == 0
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.kv_proj = nn.Linear(embed_dim, embed_dim * 2, bias=False)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query: torch.Tensor, key_value: torch.Tensor) -> torch.Tensor:
        """
        Args:
            query: (B, N_q, C) - Search patches
            key_value: (B, N_kv, C) - Reference features
            
        Returns:
            output: (B, N_q, C) - Attended search features
        """
        B, N_q, C = query.shape
        B, N_kv, C = key_value.shape
        
        # Generate Q from search, K,V from reference
        q = self.q_proj(query).reshape(B, N_q, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        kv = self.kv_proj(key_value).reshape(B, N_kv, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        
        # Cross attention: Q from search, K,V from reference
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention
        x = (attn @ v).transpose(1, 2).reshape(B, N_q, C)
        x = self.proj(x)
        
        return x


class TransformerBlock(nn.Module):
    """Transformer block with self-attention and FFN."""
    
    def __init__(self, embed_dim: int, num_heads: int = 8, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention with residual
        x = x + self.attn(self.norm1(x))
        
        # FFN with residual
        x = x + self.mlp(self.norm2(x))
        
        return x


class EnhancedSiameseDetector(nn.Module):
    """
    Enhanced Siamese Detector V2 with EfficientNet-B3 and attention mechanisms.
    
    Fixed Configuration:
    - Input: 640x640
    - Backbone: EfficientNet-B3
    - Embed dim: 512
    - Patches: 8 (2x4 grid)
    
    Architecture:
    1. Reference image -> EfficientNet -> Global features
    2. Search image -> EfficientNet -> Split into 8 patches -> Self-attention
    3. Cross-attention: Search patches (Q) x Reference features (K,V)
    4. Detection heads: Classification + Regression per patch
    """
    
    def __init__(
        self,
        num_heads: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Fixed configuration
        self.img_size = 640
        self.embed_dim = 512
        self.num_patches = 16
        
        # Shared EfficientNet-B3 backbone
        self.backbone = build_efficientnet_backbone(
            pretrained=True,
            out_channels=self.embed_dim
        )
        
        # Patch embedding for search images
        self.patch_embed = PatchEmbedding(embed_dim=self.embed_dim)
        
        # Reference feature processing - keep spatial tokens instead of global pooling
        self.ref_spatial_pool = nn.AdaptiveAvgPool2d((4, 4))  # Keep 4x4 spatial tokens
        self.ref_spatial_norm = nn.LayerNorm(self.embed_dim)
        
        # Self-attention layers for search patches
        self.self_attn_layers = nn.ModuleList([
            TransformerBlock(self.embed_dim, num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])
        
        # Cross-attention layer
        self.cross_attn = CrossAttention(self.embed_dim, num_heads, dropout)
        
        # Final processing
        self.final_norm = nn.LayerNorm(self.embed_dim)
        
        # Enhanced detection heads with conv refinement
        # First reshape patches back to 4x4 spatial layout for conv processing
        self.spatial_refine = nn.Sequential(
            # Patch tokens -> 4x4 spatial map
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
            nn.ReLU(inplace=True)
        )
        
        # Conv refinement on 4x4 spatial layout
        self.conv_refine = nn.Sequential(
            nn.Conv2d(self.embed_dim, self.embed_dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.embed_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.embed_dim // 2, self.embed_dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.embed_dim // 2),
            nn.ReLU(inplace=True)
        )
        
        # Classification head with smooth activation
        self.cls_head = nn.Sequential(
            nn.Flatten(),  # Flatten 4x4 spatial
            nn.Linear((self.embed_dim // 2) * 16, self.embed_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(self.embed_dim // 2, self.embed_dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dim // 4, 16),  # 16 patches
            nn.Sigmoid()  # Smooth activation instead of raw logits
        )
        
        # Regression head
        self.reg_head = nn.Sequential(
            nn.Flatten(),  # Flatten 4x4 spatial
            nn.Linear((self.embed_dim // 2) * 16, self.embed_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(self.embed_dim // 2, self.embed_dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dim // 4, 16 * 4)  # 16 patches × 4 coords
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features using shared backbone."""
        return self.backbone(x)
    
    def forward(self, template: torch.Tensor, search: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            template: (B, 3, 640, 640) - Reference images
            search: (B, 3, 640, 640) - Search images
            
        Returns:
            cls_logits: (B, 8, 1) - Classification scores per patch
            bbox_deltas: (B, 8, 4) - Bbox deltas per patch
        """
        B = template.shape[0]
        
        # Extract features from both images
        template_feat = self.extract_features(template)  # (B, 512, H_t, W_t)
        search_feat = self.extract_features(search)      # (B, 512, H_s, W_s)
        
        # Process reference: Keep spatial tokens
        ref_spatial = self.ref_spatial_pool(template_feat)  # (B, 512, 4, 4)
        ref_tokens = ref_spatial.flatten(2).permute(0, 2, 1)  # (B, 16, 512)
        ref_tokens = self.ref_spatial_norm(ref_tokens)  # (B, 16, 512)
        
        # Process search: Patch embedding
        search_patches = self.patch_embed(search_feat)  # (B, 16, 512)
        
        # Self-attention between search patches
        for layer in self.self_attn_layers:
            search_patches = layer(search_patches)
        
        # Cross-attention: Search patches attend to reference spatial tokens
        attended_patches = self.cross_attn(search_patches, ref_tokens)  # (B, 16, 512)
        
        # Final normalization
        attended_patches = self.final_norm(attended_patches)
        
        # Enhanced detection heads with spatial processing
        # Refine features
        refined_patches = self.spatial_refine(attended_patches)  # (B, 16, 512)
        
        # Reshape to 4x4 spatial layout for conv processing
        B = refined_patches.shape[0]
        spatial_features = refined_patches.view(B, 4, 4, self.embed_dim).permute(0, 3, 1, 2)  # (B, 512, 4, 4)
        
        # Conv refinement
        conv_features = self.conv_refine(spatial_features)  # (B, 256, 4, 4)
        
        # Detection heads
        cls_probs = self.cls_head(conv_features)    # (B, 16) - Already sigmoid activated
        bbox_deltas = self.reg_head(conv_features)  # (B, 64) -> reshape to (B, 16, 4)
        bbox_deltas = bbox_deltas.view(B, 16, 4)
        
        # Reshape cls_probs to match expected format
        cls_probs = cls_probs.unsqueeze(-1)  # (B, 16, 1)
        
        return cls_probs, bbox_deltas
    
    def get_patch_grid_info(self):
        """Get patch grid information for post-processing."""
        return {
            'grid_h': 4,
            'grid_w': 4,
            'num_patches': 16,
            'img_size': 640
        }


def count_parameters(model, verbose=False):
    """Count model parameters with detailed breakdown."""
    total_params = 0
    trainable_params = 0
    
    if verbose:
        print("\n" + "="*80)
        print("MODEL PARAMETER BREAKDOWN")
        print("="*80)
    
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf modules only
            module_params = sum(p.numel() for p in module.parameters())
            module_trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
            
            if module_params > 0 and verbose:
                print(f"{name:50} {module_params:>12,} ({module_params/1e6:>6.2f}M)")
            
            total_params += module_params
            trainable_params += module_trainable
    
    if verbose:
        print("="*80)
        print(f"{'TOTAL PARAMETERS':50} {total_params:>12,} ({total_params/1e6:>6.2f}M)")
        print(f"{'TRAINABLE PARAMETERS':50} {trainable_params:>12,} ({trainable_params/1e6:>6.2f}M)")
        print("="*80)
    
    return total_params, trainable_params


if __name__ == "__main__":
    # Quick test
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EnhancedSiameseDetector(num_heads=8, num_layers=4, dropout=0.1).to(device)
    template = torch.randn(2, 3, 640, 640).to(device)
    search = torch.randn(2, 3, 640, 640).to(device)
    with torch.no_grad():
        cls_logits, bbox_deltas = model(template, search)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model test passed: {total_params/1e6:.2f}M params, output shapes: {cls_logits.shape}, {bbox_deltas.shape}")
