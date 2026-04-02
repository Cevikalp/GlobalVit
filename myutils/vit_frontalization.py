
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from sklearn.decomposition import PCA
import torch
import numpy as np
import matplotlib.pyplot as plt
from backbones import get_model 
from transformers import ViTModel, ViTConfig

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(CNNBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, 4, stride, 1, bias=False, padding_mode="reflect"
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.conv(x)

class Discriminator(nn.Module):
    def __init__(self, in_channels=3, features=[32, 64, 128, 256]):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(
                in_channels * 2,
                features[0],
                kernel_size=4,
                stride=2,
                padding=1,
                padding_mode="reflect",
            ),
            nn.LeakyReLU(0.2),
        )

        layers = []
        in_channels = features[0]
        for feature in features[1:]:
            layers.append(
                CNNBlock(in_channels, feature, stride=1 if feature == features[-1] else 2),
            )
            in_channels = feature

        layers.append(
            nn.Conv2d(
                in_channels, 1, kernel_size=4, stride=1, padding=1, padding_mode="reflect"
            ),
        )

        self.model = nn.Sequential(*layers)

    def forward(self, x, y):
        x = torch.cat([x, y], dim=1)
        x = self.initial(x)
        x = self.model(x)
        return x



# ==============================================================================
# 1. Corrected Fixed2DPositionalEncoding Class
# ==============================================================================
class Fixed2DPositionalEncoding(nn.Module):
    """
    Corrected 2D sinusoidal positional encoding that utilizes the full embedding dimension.
    """
    def __init__(self, embed_dim, grid_size):
        super().__init__()
        # Ensure the embedding dimension is even, as it will be split for X and Y axes.
        assert embed_dim % 2 == 0, "Embedding dimension must be an even number."
        
        self.embed_dim = embed_dim
        self.grid_size = grid_size  # Expected format: (H, W)
        
        # 'register_buffer' makes this a persistent part of the model that is not trained.
        self.register_buffer('pos_embed', self._create_sine_positional_encoding())

    def _create_sine_positional_encoding(self):
        H, W = self.grid_size
        # Use half of the embedding dimension for each axis (X and Y).
        half_dim = self.embed_dim // 2
        
        # Create the division term based on the original Transformer paper's formula.
        div_term = torch.exp(torch.arange(0, half_dim, 2).float() * -(math.log(10000.0) / half_dim))

        # Create coordinate grids
        y_pos = torch.arange(H, dtype=torch.float32).unsqueeze(1) # Shape: [H, 1]
        x_pos = torch.arange(W, dtype=torch.float32).unsqueeze(0) # Shape: [1, W]

        # Calculate sine/cosine embeddings for the X-axis
        pe_x = torch.zeros(H, W, half_dim, dtype=torch.float32)
        pe_x[:, :, 0::2] = torch.sin(x_pos.unsqueeze(-1) * div_term)
        pe_x[:, :, 1::2] = torch.cos(x_pos.unsqueeze(-1) * div_term)
        
        # Calculate sine/cosine embeddings for the Y-axis
        pe_y = torch.zeros(H, W, half_dim, dtype=torch.float32)
        pe_y[:, :, 0::2] = torch.sin(y_pos.unsqueeze(-1) * div_term)
        pe_y[:, :, 1::2] = torch.cos(y_pos.unsqueeze(-1) * div_term)
        
        # Concatenate the X and Y embeddings to form the full positional embedding.
        pe = torch.cat([pe_x, pe_y], dim=-1) # Shape: [H, W, embed_dim]
        
        # Reshape for use with patch sequences.
        pe = pe.view(1, H * W, self.embed_dim) # Shape: [1, num_patches, embed_dim]
        return pe

    def forward(self, x):
        # x shape: [Batch, num_patches, embed_dim]
        # self.pos_embed shape: [1, num_patches, embed_dim]
        # Broadcasting adds the positional encoding to each item in the batch.
        return x + self.pos_embed

class Learnable2DPositionalEncoding(nn.Module):
    """
    Learnable 2D positional encoding for ViT-style models.

    Each spatial location gets a learnable embedding vector, similar to ViT.
    """
    def __init__(self, embed_dim, grid_size):
        """
        Args:
            embed_dim (int): Embedding dimension (must match patch embedding dim).
            grid_size (tuple): Tuple (H, W) — number of patches along height and width.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.grid_size = grid_size  # (H, W)
        H, W = grid_size

        # Create learnable positional embeddings of shape (H*W, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, H * W, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)  # recommended init

    def forward(self, x):
        """
        Args:
            x (Tensor): Shape [B, N, D] — patch sequence
        Returns:
            Tensor: [B, N, D] with positional encoding added
        """
        return x + self.pos_embed  # 

# ==============================================================================
# 2. Corrected ViTFrontalizationEncoderDecoder Class
# ==============================================================================
class ViTFrontalizationEncoderDecoder(nn.Module):
    def __init__(self, img_size=256, patch_size=16, embed_dim=768, depth=6, num_heads=8, decoder_depth=3):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        assert img_size % patch_size == 0, "Image size must be divisible by patch size."
        
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_dim = 3 * patch_size * patch_size
        grid_size = img_size // patch_size

        # Use nn.Unfold for patch extraction
        self.unfold = nn.Unfold(kernel_size=patch_size, stride=patch_size)
        self.fold = nn.Fold(output_size=(img_size, img_size), kernel_size=patch_size, stride=patch_size)

        self.patch_embed = nn.Linear(self.patch_dim, embed_dim)

        self.pos_encoding = Learnable2DPositionalEncoding(embed_dim, (grid_size, grid_size))
        self.pre_transformer_norm = nn.LayerNorm(embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)

       # self.decoder_queries = nn.Parameter(torch.randn(self.num_patches, embed_dim))


        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=num_heads)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=decoder_depth)

        self.output_proj = nn.Linear(embed_dim, self.patch_dim)
        # ADD a convolutional head
        self.decoder_head = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, embed_dim // 2, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.GroupNorm(8, embed_dim // 2),
            nn.ReLU(),
            nn.ConvTranspose2d(embed_dim // 2, embed_dim // 4, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.GroupNorm(8, embed_dim // 4),
            nn.ReLU(),
           # nn.ConvTranspose2d(embed_dim // 4, embed_dim // 8, kernel_size=3, stride=2, padding=1, output_padding=1),
           # nn.ReLU(),
          #  nn.ConvTranspose2d(embed_dim // 8, embed_dim // 8, kernel_size=3, stride=2, padding=1, output_padding=1),
           # nn.ReLU(),
            # This last layer gets you back to the original image size and 3 channels
            nn.ConvTranspose2d(embed_dim // 4, 3, kernel_size=3, stride=2, padding=1, output_padding=1)
        )
        self.final_activation = nn.Sigmoid()

    def forward(self, x):
       
        B, C, H, W = x.shape
        assert H == self.img_size and W == self.img_size, "Input image size mismatch."

        patches = self.unfold(x)                      # [B, patch_dim, N]
        patches = patches.transpose(1, 2)             # [B, N, patch_dim]

        x = self.patch_embed(patches)                 # [B, N, embed_dim]
        x = self.pos_encoding(x)                      # Add positional encoding
        x = self.pre_transformer_norm(x)              # Optional pre-norm

        x = x.transpose(0, 1)                         # [N, B, D]
        memory = self.encoder(x)                      # [N, B, D], this is the encoded input

        # --- FIX IS HERE ---
        # The decoder's target should be the encoded input itself.
        # This forces the decoder to reconstruct based on the specific input.
        # We no longer need `self.decoder_queries`.
        decoded = self.decoder(tgt=memory, memory=memory) # [N, B, D]
        # --- END OF FIX ---

        decoded = decoded.transpose(0, 1)             # [B, N, D]

        # previous
       # projected = self.output_proj(decoded)         # [B, N, patch_dim]
       # projected = projected.transpose(1, 2)         # [B, patch_dim, N]
       # output = self.fold(projected)                 # [B, 3, H, W]


        # --- NEW DECODER HEAD LOGIC ---
        B, N, D = decoded.shape
        grid_h = grid_w = self.img_size // self.patch_size
        feature_map = decoded.transpose(1, 2).view(B, D, grid_h, grid_w)
        # Reshape from a sequence of patches to a 2D feature map
       # feature_map = decoded.transpose(1, 2).view(B, D, grid_size, grid_size) # [B, D, H', W']

        # Upsample using the convolutional head
        output = self.decoder_head(feature_map) # [B, 3, H, W]
       # output = self.final_activation(output)  # Apply final activation (e.g., Sigmoid for pixel values)

        return output

class ViTFrontalizationEncoder(nn.Module):
    def __init__(self, img_size=256, patch_size=16, embed_dim=768, depth=6, num_heads=8, decoder_depth=3):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        assert img_size % patch_size == 0, "Image size must be divisible by patch size."
        
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_dim = 3 * patch_size * patch_size
        grid_size = img_size // patch_size

        # Use nn.Unfold for patch extraction
        self.unfold = nn.Unfold(kernel_size=patch_size, stride=patch_size)
        self.fold = nn.Fold(output_size=(img_size, img_size), kernel_size=patch_size, stride=patch_size)

        self.patch_embed = nn.Linear(self.patch_dim, embed_dim)

        self.pos_encoding = Learnable2DPositionalEncoding(embed_dim, (grid_size, grid_size))
        self.pre_transformer_norm = nn.LayerNorm(embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)

       # self.decoder_queries = nn.Parameter(torch.randn(self.num_patches, embed_dim))


        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=num_heads)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=decoder_depth)

        self.output_proj = nn.Linear(embed_dim, self.patch_dim)
        # ADD a convolutional head
        self.decoder_head = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, embed_dim // 2, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.GroupNorm(8, embed_dim // 2),
            nn.ReLU(),
            nn.ConvTranspose2d(embed_dim // 2, embed_dim // 4, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.GroupNorm(8, embed_dim // 4),
            nn.ReLU(),
           # nn.ConvTranspose2d(embed_dim // 4, embed_dim // 8, kernel_size=3, stride=2, padding=1, output_padding=1),
           # nn.ReLU(),
          #  nn.ConvTranspose2d(embed_dim // 8, embed_dim // 8, kernel_size=3, stride=2, padding=1, output_padding=1),
           # nn.ReLU(),
            # This last layer gets you back to the original image size and 3 channels
            nn.ConvTranspose2d(embed_dim // 4, 3, kernel_size=3, stride=2, padding=1, output_padding=1)
        )
        self.final_activation = nn.Sigmoid()

    def forward(self, x):
       
        B, C, H, W = x.shape
        assert H == self.img_size and W == self.img_size, "Input image size mismatch."

        patches = self.unfold(x)                      # [B, patch_dim, N]
        patches = patches.transpose(1, 2)             # [B, N, patch_dim]

        x = self.patch_embed(patches)                 # [B, N, embed_dim]
        x = self.pos_encoding(x)                      # Add positional encoding
        x = self.pre_transformer_norm(x)              # Optional pre-norm

        x = x.transpose(0, 1)                         # [N, B, D]
        memory = self.encoder(x)                      # [N, B, D], this is the encoded input

        # --- FIX IS HERE ---
        # The decoder's target should be the encoded input itself.
        # This forces the decoder to reconstruct based on the specific input.
        # We no longer need `self.decoder_queries`.
        #decoded = self.decoder(tgt=memory, memory=memory) # [N, B, D]
        # --- END OF FIX ---
        decoded = memory
        decoded = decoded.transpose(0, 1)             # [B, N, D]

        # previous
       # projected = self.output_proj(decoded)         # [B, N, patch_dim]
       # projected = projected.transpose(1, 2)         # [B, patch_dim, N]
       # output = self.fold(projected)                 # [B, 3, H, W]


        # --- NEW DECODER HEAD LOGIC ---
        B, N, D = decoded.shape
        grid_h = grid_w = self.img_size // self.patch_size
        feature_map = decoded.transpose(1, 2).view(B, D, grid_h, grid_w)
        # Reshape from a sequence of patches to a 2D feature map
       # feature_map = decoded.transpose(1, 2).view(B, D, grid_size, grid_size) # [B, D, H', W']

        # Upsample using the convolutional head
        output = self.decoder_head(feature_map) # [B, 3, H, W]
       # output = self.final_activation(output)  # Apply final activation (e.g., Sigmoid for pixel values)

        return output


class ViTFrontalizationEncoderDecoderSmall(nn.Module):
    def __init__(self, img_size=128, patch_size=8, embed_dim=768, depth=6, num_heads=8, decoder_depth=3):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        assert img_size % patch_size == 0, "Image size must be divisible by patch size."
        
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_dim = 3 * patch_size * patch_size
        grid_size = img_size // patch_size

        # Use nn.Unfold for patch extraction
        self.unfold = nn.Unfold(kernel_size=patch_size, stride=patch_size)
        self.fold = nn.Fold(output_size=(img_size, img_size), kernel_size=patch_size, stride=patch_size)

        self.patch_embed = nn.Linear(self.patch_dim, embed_dim)

        self.pos_encoding = Learnable2DPositionalEncoding(embed_dim, (grid_size, grid_size))
        self.pre_transformer_norm = nn.LayerNorm(embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)

       # self.decoder_queries = nn.Parameter(torch.randn(self.num_patches, embed_dim))


        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=num_heads)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=decoder_depth)

        self.output_proj = nn.Linear(embed_dim, self.patch_dim)
        # ADD a convolutional head
        self.decoder_head = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, embed_dim // 2, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.GroupNorm(8, embed_dim // 2),
            nn.ReLU(),
            nn.ConvTranspose2d(embed_dim // 2, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
         #   nn.GroupNorm(8, embed_dim // 4),
         #   nn.ReLU(),
           # nn.ConvTranspose2d(embed_dim // 4, embed_dim // 8, kernel_size=3, stride=2, padding=1, output_padding=1),
           # nn.ReLU(),
          #  nn.ConvTranspose2d(embed_dim // 8, embed_dim // 8, kernel_size=3, stride=2, padding=1, output_padding=1),
           # nn.ReLU(),
            # This last layer gets you back to the original image size and 3 channels
          #  nn.ConvTranspose2d(embed_dim // 4, 3, kernel_size=3, stride=2, padding=1, output_padding=1)
        )
        self.final_activation = nn.Sigmoid()

    def forward(self, x):
       
        B, C, H, W = x.shape
        assert H == self.img_size and W == self.img_size, "Input image size mismatch."

        patches = self.unfold(x)                      # [B, patch_dim, N]
        patches = patches.transpose(1, 2)             # [B, N, patch_dim]

        x = self.patch_embed(patches)                 # [B, N, embed_dim]
        x = self.pos_encoding(x)                      # Add positional encoding
        x = self.pre_transformer_norm(x)              # Optional pre-norm

        x = x.transpose(0, 1)                         # [N, B, D]
        memory = self.encoder(x)                      # [N, B, D], this is the encoded input

        # --- FIX IS HERE ---
        # The decoder's target should be the encoded input itself.
        # This forces the decoder to reconstruct based on the specific input.
        # We no longer need `self.decoder_queries`.
        decoded = self.decoder(tgt=memory, memory=memory) # [N, B, D]
        # --- END OF FIX ---

        decoded = decoded.transpose(0, 1)             # [B, N, D]

        # previous
       # projected = self.output_proj(decoded)         # [B, N, patch_dim]
       # projected = projected.transpose(1, 2)         # [B, patch_dim, N]
       # output = self.fold(projected)                 # [B, 3, H, W]


        # --- NEW DECODER HEAD LOGIC ---
        B, N, D = decoded.shape
        grid_h = grid_w = self.img_size // self.patch_size
        feature_map = decoded.transpose(1, 2).view(B, D, grid_h, grid_w)
        # Reshape from a sequence of patches to a 2D feature map
       # feature_map = decoded.transpose(1, 2).view(B, D, grid_size, grid_size) # [B, D, H', W']

        # Upsample using the convolutional head
        output = self.decoder_head(feature_map) # [B, 3, H, W]
       # output = self.final_activation(output)  # Apply final activation (e.g., Sigmoid for pixel values)

        return output

class ViTFrontalizationEncoderDecoderDETR(nn.Module):
    def __init__(self, img_size=256, patch_size=16, embed_dim=768, depth=6, num_heads=8, decoder_depth=3):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        assert img_size % patch_size == 0, "Image size must be divisible by patch size."
        
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_dim = 3 * patch_size * patch_size
        grid_size = img_size // patch_size

        # Patch extraction & reconstruction
        self.unfold = nn.Unfold(kernel_size=patch_size, stride=patch_size)
        self.fold = nn.Fold(output_size=(img_size, img_size), kernel_size=patch_size, stride=patch_size)

        # Patch embedding
        self.patch_embed = nn.Linear(self.patch_dim, embed_dim)

        # Positional encoding
        self.pos_encoding = Learnable2DPositionalEncoding(embed_dim, (grid_size, grid_size))
        self.pre_transformer_norm = nn.LayerNorm(embed_dim)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dropout=0.1)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        # --- Learnable decoder queries (like DETR) ---
        self.decoder_queries = nn.Parameter(torch.randn(self.num_patches, embed_dim))  # [N, D]

        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=num_heads, dropout=0.1)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=decoder_depth)

        # CNN decoder head for upsampling
        self.decoder_head = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, embed_dim // 2, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.GroupNorm(8, embed_dim // 2),
            nn.ReLU(),
            nn.ConvTranspose2d(embed_dim // 2, embed_dim // 4, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.GroupNorm(8, embed_dim // 4),
            nn.ReLU(),
            nn.ConvTranspose2d(embed_dim // 4, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
        )
        self.final_activation = nn.Sigmoid()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size and W == self.img_size, "Input image size mismatch."

        # Extract patches
        patches = self.unfold(x)                      # [B, patch_dim, N]
        patches = patches.transpose(1, 2)             # [B, N, patch_dim]

        # Embed patches + pos encoding
        x = self.patch_embed(patches)                 # [B, N, D]
        x = self.pos_encoding(x)                      # [B, N, D]
        x = self.pre_transformer_norm(x)              # [B, N, D]

        # Transformer encoder
        x = x.transpose(0, 1)                         # [N, B, D]
        memory = self.encoder(x)                      # [N, B, D]

        # Expand learnable queries for batch
        queries = self.decoder_queries.unsqueeze(1).repeat(1, B, 1)  # [N, B, D]

        # Transformer decoder
        decoded = self.decoder(tgt=queries, memory=memory)  # [N, B, D]
        decoded = decoded.transpose(0, 1)             # [B, N, D]

        # Reshape into feature map
        B, N, D = decoded.shape
        grid_h = grid_w = self.img_size // self.patch_size
        feature_map = decoded.transpose(1, 2).view(B, D, grid_h, grid_w)

        # CNN head to image
        output = self.decoder_head(feature_map)       # [B, 3, H, W]
        output = self.final_activation(output)
        return output

class ViTFrontalizationEncoderDecoderLast(nn.Module):
    def __init__(self, img_size=256, patch_size=16, embed_dim=768, depth=6, num_heads=8, num_classes=2, decoder_depth=3):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_classes = num_classes

        assert img_size % patch_size == 0, "Image size must be divisible by patch size."
        
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_dim = 3 * patch_size * patch_size
        grid_size = img_size // patch_size

        # Use nn.Unfold for patch extraction
        self.unfold = nn.Unfold(kernel_size=patch_size, stride=patch_size)
        self.fold = nn.Fold(output_size=(img_size, img_size), kernel_size=patch_size, stride=patch_size)

        self.patch_embed = nn.Linear(self.patch_dim, embed_dim)

        self.pos_encoding = Learnable2DPositionalEncoding(embed_dim, (grid_size, grid_size))
        #self.pos_encoding = Fixed2DPositionalEncoding(embed_dim, (grid_size, grid_size))
        self.pre_transformer_norm = nn.LayerNorm(embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)

       # self.decoder_queries = nn.Parameter(torch.randn(self.num_patches, embed_dim))


        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=num_heads)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=decoder_depth)

        self.output_proj = nn.Linear(embed_dim, self.patch_dim)
        # ADD a convolutional head
        self.decoder_head = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, embed_dim // 2, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.GroupNorm(8, embed_dim // 2),
            nn.ReLU(),
            nn.ConvTranspose2d(embed_dim // 2, embed_dim // 4, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.GroupNorm(8, embed_dim // 4),
            nn.ReLU(),
           # nn.ConvTranspose2d(embed_dim // 4, embed_dim // 8, kernel_size=3, stride=2, padding=1, output_padding=1),
           # nn.GroupNorm(8, embed_dim // 8),
           # nn.ReLU(),
          
          #  nn.ConvTranspose2d(embed_dim // 8, embed_dim // 8, kernel_size=3, stride=2, padding=1, output_padding=1),
           # nn.ReLU(),
            # This last layer gets you back to the original image size and 3 channels
            nn.ConvTranspose2d(embed_dim // 4, 3, kernel_size=3, stride=2, padding=1, output_padding=1)
        )
        self.final_activation = nn.Sigmoid()

    def forward(self, x):
       
        B, C, H, W = x.shape
        assert H == self.img_size and W == self.img_size, "Input image size mismatch."

        patches = self.unfold(x)                      # [B, patch_dim, N]
        patches = patches.transpose(1, 2)             # [B, N, patch_dim]

        x = self.patch_embed(patches)                 # [B, N, embed_dim]
        x = self.pos_encoding(x)                      # Add positional encoding
        x = self.pre_transformer_norm(x)              # Optional pre-norm

        x = x.transpose(0, 1)                         # [N, B, D]
        memory = self.encoder(x)                      # [N, B, D], this is the encoded input

        # --- FIX IS HERE ---
        # The decoder's target should be the encoded input itself.
        # This forces the decoder to reconstruct based on the specific input.
        # We no longer need `self.decoder_queries`.
        #memory = self.pos_encoding(memory.transpose(1,0)).transpose(0,1)  # Add positional encoding to memory
        queries = self.pos_encoding(memory.transpose(1,0)).transpose(0,1)
        
        decoded = self.decoder(tgt=queries, memory=memory, tgt_mask = None, memory_mask= None, tgt_key_padding_mask=None,
                               memory_key_padding_mask = None)

        #decoded = self.decoder(tgt=memory, memory=memory) # [N, B, D]
        # --- END OF FIX ---

        decoded = decoded.transpose(0, 1)             # [B, N, D]

        # previous
       # projected = self.output_proj(decoded)         # [B, N, patch_dim]
       # projected = projected.transpose(1, 2)         # [B, patch_dim, N]
       # output = self.fold(projected)                 # [B, 3, H, W]


        # --- NEW DECODER HEAD LOGIC ---
        B, N, D = decoded.shape
        grid_h = grid_w = self.img_size // self.patch_size
        feature_map = decoded.transpose(1, 2).view(B, D, grid_h, grid_w)
        # Reshape from a sequence of patches to a 2D feature map
       # feature_map = decoded.transpose(1, 2).view(B, D, grid_size, grid_size) # [B, D, H', W']

        # Upsample using the convolutional head
        output = self.decoder_head(feature_map) # [B, 3, H, W]
       # output = self.final_activation(output)  # Apply final activation (e.g., Sigmoid for pixel values)
        if output.shape[-2:] != (H, W):
            output = F.interpolate(output, size=(H, W), mode='bilinear', align_corners=False)

        return output


class FaRLFrontalizationEncoderDecoder(nn.Module):
    def __init__(self, weight_path=None, img_size=256, embed_dim=768, num_heads=12, decoder_depth=3):
        super().__init__()

        self.img_size = img_size
        self.embed_dim = embed_dim
        self.encoder_grid = 16 
        self.num_patches = self.encoder_grid ** 2

        weight_path = '/media/mlcv/SSD/face_transformer_model/FaRL-Base-Patch16-LAIONFace20M-ep64.pth'

        # 1. Setup Architecture
        config = ViTConfig.from_pretrained("google/vit-base-patch16-224-in21k")
        config.image_size = img_size 
        self.encoder = ViTModel(config)
        
        # 2. Inject FaRL Weights
        if weight_path:
            state_dict = torch.load(weight_path, map_location='cpu')
            
            # FaRL weights often come inside a 'state_dict' key or with a prefix
            if 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            
            # Clean prefixes if they exist (common in FaRL/CLIP checkpoints)
            new_state_dict = {}
            for k, v in state_dict.items():
                name = k.replace('backbone.', '') # Adjust this based on your .pth file
                new_state_dict[name] = v
                
            msg = self.encoder.load_state_dict(new_state_dict, strict=False)
            print(f"Loaded FaRL weights from {weight_path}. Missing keys: {len(msg.missing_keys)}")

        # 3. Decoder Positional Embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # 4. Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=decoder_depth)

        # 5. Upsampling Head
        self.decoder_head = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, 256, 4, 2, 1), # 16 -> 32
            nn.GroupNorm(32, 256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(256, 128, 4, 2, 1),      # 32 -> 64
            nn.GroupNorm(16, 128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(128, 64, 4, 2, 1),       # 64 -> 128
            nn.GroupNorm(8, 64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(64, 3, 4, 2, 1),         # 128 -> 256
            nn.Tanh()
        )

    def forward(self, x):
        B = x.shape[0]

        # 1. Extract features
        outputs = self.encoder(x)
        
        # 2. Extract spatial tokens [B, 256, 768] 
        # (Index 1: skips the CLS token)
        memory = outputs.last_hidden_state[:, 1:, :]

        # 3. Decode
        queries = memory + self.pos_embed
        decoded = self.decoder(tgt=queries, memory=memory)

        # 4. Reshape to [B, 768, 16, 16]
        feature_map = decoded.transpose(1, 2).reshape(
            B, self.embed_dim, self.encoder_grid, self.encoder_grid
        )

        # 5. Generate image
        return self.decoder_head(feature_map)


class ViTFrontalizationEncoderDecoderPretrained(nn.Module):
    def __init__(self, img_size=256, embed_dim=512, num_heads=8, decoder_depth=3):
        super().__init__()

        self.img_size = img_size
        self.embed_dim = embed_dim
        
        # TransFace-B uses a 112x112 input with patch_size=9
        # This results in a 12x12 grid (144 patches)
        self.encoder_grid = 12 
        self.num_patches = self.encoder_grid ** 2

        # 1. Pretrained TransFace Encoder
        # We assume get_model returns the VisionTransformer class you shared
        self.encoder = get_model('vit_b', fp16=False)
        self.encoder.load_state_dict(
            torch.load('/media/mlcv/Data/TransformerFrontalization/face_models/vit_B.pt')
        )
        
        # Freeze encoder for identity preservation
        #for p in self.encoder.parameters():
         #   p.requires_grad = False
        #self.encoder.eval()

        # 2. Decoder Positional Embedding (Specific to our 12x12 grid)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # 3. Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            batch_first=True # Simplifies code; no more .transpose(0, 1)
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=decoder_depth)

        # 4. Upsampling Head (12x12 -> 256x256)
        # We start with 12x12. Interp to 16x16, then 4 doublings = 256.
        self.decoder_head = nn.Sequential(
            nn.Upsample(size=(16, 16), mode='bilinear', align_corners=False),
            
            # 16x16 -> 32x32
            nn.ConvTranspose2d(embed_dim, 256, 4, 2, 1),
            nn.GroupNorm(32, 256),
            nn.LeakyReLU(0.2, inplace=True),

            # 32x32 -> 64x64
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.GroupNorm(16, 128),
            nn.LeakyReLU(0.2, inplace=True),

            # 64x64 -> 128x128
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.GroupNorm(8, 64),
            nn.LeakyReLU(0.2, inplace=True),

            # 128x128 -> 256x256
            nn.ConvTranspose2d(64, 3, 4, 2, 1),
            nn.Tanh() # Assuming output range [-1, 1]
        )

    def extract_encoder_features(self, x):
        """Extracts raw (B, 144, 512) tokens from TransFace without SE-Net masking."""
        B = x.shape[0]
        # 1. Patching & Positional Embedding
        x = self.encoder.patch_embed(x)
        x = x + self.encoder.pos_embed
        x = self.encoder.pos_drop(x)

        # 2. Transformer Blocks (Bypassing mask_ratio logic for stability)
        for block in self.encoder.blocks:
            x = block(x)
        
        # 3. Final Norm
        x = self.encoder.norm(x)
        return x

    def forward(self, x):
        B = x.shape[0]

        # 1. TransFace expects 112x112 input
        x_112 = F.interpolate(x, size=(112, 112), mode='bilinear', align_corners=False)

        # 2. Extract spatial tokens (No CLS token, no SE-Net weights)
        #with torch.no_grad():
        memory = self.extract_encoder_features(x_112)

        # 3. Decode: Add learnable positions to the queries
        queries = memory + self.pos_embed
        decoded = self.decoder(tgt=queries, memory=memory)

        # 4. Reshape tokens -> Feature Map (B, 512, 12, 12)
        feature_map = decoded.transpose(1, 2).view(
            B, self.embed_dim, self.encoder_grid, self.encoder_grid
        )

        # 5. Generate 256x256 image
        return self.decoder_head(feature_map)

class ViTFrontalizationEncoderDecoderNew(nn.Module):
    def __init__(self, img_size=256, patch_size=16, embed_dim=768, depth=6, num_heads=8, decoder_depth=3):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        assert img_size % patch_size == 0, "Image size must be divisible by patch size."
        
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_dim = 3 * patch_size * patch_size
        grid_size = img_size // patch_size

        # Patch extraction & reconstruction
        self.unfold = nn.Unfold(kernel_size=patch_size, stride=patch_size)
        self.fold = nn.Fold(output_size=(img_size, img_size), kernel_size=patch_size, stride=patch_size)

        # Patch embedding
        self.patch_embed = nn.Linear(self.patch_dim, embed_dim)

        # Positional encoding
        self.pos_encoding = Learnable2DPositionalEncoding(embed_dim, (grid_size, grid_size))
        self.pre_transformer_norm = nn.LayerNorm(embed_dim)

        # Transformer encoder (default: seq_first = [N, B, D])
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dropout=0.1
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        # Transformer decoder (also seq_first)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim, nhead=num_heads, dropout=0.1
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=decoder_depth)

        # Projection back to patches (if needed)
        self.output_proj = nn.Linear(embed_dim, self.patch_dim)

        # CNN decoder head for upsampling
        self.decoder_head = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, embed_dim // 2, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.GroupNorm(8, embed_dim // 2),
            nn.ReLU(),
            nn.ConvTranspose2d(embed_dim // 2, embed_dim // 4, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.GroupNorm(8, embed_dim // 4),
            nn.ReLU(),
            nn.ConvTranspose2d(embed_dim // 4, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
        )
        self.final_activation = nn.Sigmoid()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size and W == self.img_size, "Input image size mismatch."

        # Extract patches
        patches = self.unfold(x)                      # [B, patch_dim, N]
        patches = patches.transpose(1, 2)             # [B, N, patch_dim]

        # Linear embedding + positional encoding
        x = self.patch_embed(patches)                 # [B, N, D]
        x = self.pos_encoding(x)                      # [B, N, D]
        x = self.pre_transformer_norm(x)              # [B, N, D]

        # Transformer expects [N, B, D]
        x = x.transpose(0, 1)                         # [N, B, D]
        memory = self.encoder(x)                      # [N, B, D]

        decoded = self.decoder(tgt=memory, memory=memory)  # [N, B, D]
        decoded = decoded.transpose(0, 1)             # [B, N, D]

        # Reshape into feature map
        B, N, D = decoded.shape
        grid_h = grid_w = self.img_size // self.patch_size
        feature_map = decoded.transpose(1, 2).view(B, D, grid_h, grid_w)

        # Decode into image
        output = self.decoder_head(feature_map)       # [B, 3, H, W]
        output = self.final_activation(output)
        return output

class ViTFrontalizationEncoderDecoderNew(nn.Module):
    def __init__(self, img_size=256, patch_size=16, embed_dim=768, depth=6, num_heads=8, decoder_depth=3):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        assert img_size % patch_size == 0, "Image size must be divisible by patch size."
        
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_dim = 3 * patch_size * patch_size
        grid_size = img_size // patch_size

        # Patch extraction & reconstruction
        self.unfold = nn.Unfold(kernel_size=patch_size, stride=patch_size)
        self.fold = nn.Fold(output_size=(img_size, img_size), kernel_size=patch_size, stride=patch_size)

        # Patch embedding
        self.patch_embed = nn.Linear(self.patch_dim, embed_dim)

        # Positional encoding
        self.pos_encoding = Learnable2DPositionalEncoding(embed_dim, (grid_size, grid_size))
        self.pre_transformer_norm = nn.LayerNorm(embed_dim)

        # Transformer encoder (default: seq_first = [N, B, D])
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dropout=0.1
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        # Transformer decoder (also seq_first)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim, nhead=num_heads, dropout=0.1
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=decoder_depth)

        # Projection back to patches (if needed)
        self.output_proj = nn.Linear(embed_dim, self.patch_dim)

        # CNN decoder head for upsampling
        self.decoder_head = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, embed_dim // 2, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.GroupNorm(8, embed_dim // 2),
            nn.ReLU(),
            nn.ConvTranspose2d(embed_dim // 2, embed_dim // 4, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.GroupNorm(8, embed_dim // 4),
            nn.ReLU(),
            nn.ConvTranspose2d(embed_dim // 4, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
        )
        self.final_activation = nn.Sigmoid()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size and W == self.img_size, "Input image size mismatch."

        # Extract patches
        patches = self.unfold(x)                      # [B, patch_dim, N]
        patches = patches.transpose(1, 2)             # [B, N, patch_dim]

        # Linear embedding + positional encoding
        x = self.patch_embed(patches)                 # [B, N, D]
        x = self.pos_encoding(x)                      # [B, N, D]
        x = self.pre_transformer_norm(x)              # [B, N, D]

        # Transformer expects [N, B, D]
        x = x.transpose(0, 1)                         # [N, B, D]
        memory = self.encoder(x)                      # [N, B, D]

        decoded = self.decoder(tgt=memory, memory=memory)  # [N, B, D]
        decoded = decoded.transpose(0, 1)             # [B, N, D]

        # Reshape into feature map
        B, N, D = decoded.shape
        grid_h = grid_w = self.img_size // self.patch_size
        feature_map = decoded.transpose(1, 2).view(B, D, grid_h, grid_w)

        # Decode into image
        output = self.decoder_head(feature_map)       # [B, 3, H, W]
        output = self.final_activation(output)
        return output



def visualize_queries(queries, grid_size=16, method='pca'):
    """
    Visualizes learned queries.
    
    Args:
        queries (Tensor): shape [N, D], e.g., [256, 768]
        grid_size (int): spatial arrangement, e.g., 16x16
        method (str): 'pca' or 'tsne'
    """
    queries = queries.detach().cpu().numpy()  # [N, D]

    if method == 'pca':
        pca = PCA(n_components=2)
        reduced = pca.fit_transform(queries)  # [N, 2]
    else:
        from sklearn.manifold import TSNE
        reduced = TSNE(n_components=2, perplexity=30).fit_transform(queries)

    plt.figure(figsize=(6, 6))
    plt.scatter(reduced[:, 0], reduced[:, 1], c=np.arange(len(queries)), cmap='viridis', s=30)
    plt.colorbar(label='Patch Index')
    plt.title(f'Learned Decoder Queries ({method.upper()})')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def visualize_queries_as_grid(queries, grid_size=16):
    """
    Reshape each query to an average activation and visualize on a 2D grid.
    """
    queries = queries.detach().cpu()  # [N, D]
    grid = queries.mean(dim=1).reshape(grid_size, grid_size)  # [grid_h, grid_w]
    plt.figure(figsize=(5, 5))
    plt.imshow(grid, cmap='viridis')
    plt.colorbar(label='Mean activation')
    plt.title('Learned Decoder Queries (Mean Activation Grid)')
    plt.axis('off')
    plt.tight_layout()
    plt.show()