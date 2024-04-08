import torch
from torch import nn
from encoder import TextEncoder, ImageEncoder

# CLIP is image-text multimodal model that uses contrastive learning to learn joint embedding space for images and text.
# It uses text and image encoder and a projection head to project the embeddings to a joint space.

class CLIP(nn.Module):
    def __init__(self, text_encoder, image_encoder, projection, projection_dim = 1028):
        super(CLIP, self).__init__()
        self.text_encoder = text_encoder
        self.image_encoder = image_encoder
        self.projection_dim = projection_dim
        self.image_projection = projection
        self.text_projection = projection

    def forward(self, image, text):
        # Use tokenizer first here
        text_embed = self.text_encoder(text)
        image_embed = self.image_encoder(image)

        text_features = self.text_projection(text_embed)
        image_features = self.image_projection(image_embed)



