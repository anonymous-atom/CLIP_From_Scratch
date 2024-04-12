import torch.nn.functional as F
import torch
from torch import nn
from encoder import TextEncoder, ImageEncoder
from projection_head import ProjectionHead


# CLIP is an image-text multimodal model that uses contrastive learning to learn joint embedding space for images and
# text.
# It uses text and image encoder and a projection head to project the embeddings to a joint space.
def cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)

    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()


class CLIP(nn.Module):
    def __init__(self, text_encoder: TextEncoder,
                 image_encoder: ImageEncoder,
                 t_projection: ProjectionHead,
                 i_projection: ProjectionHead,
                 projection_dim=512):
        super(CLIP, self).__init__()

        self.text_encoder = text_encoder
        self.image_encoder = image_encoder
        self.projection_dim = projection_dim
        self.image_projection = i_projection
        self.text_projection = t_projection

    def forward(self, image, text):
        text_embed = self.text_encoder(text)
        image_embed = self.image_encoder(image)

        text_embed = torch.sum(text_embed, dim=1)
        image_embed = torch.sum(image_embed, dim=1)

        text_features = self.text_projection(text_embed)
        image_features = self.image_projection(image_embed)

        logits = text_features @ image_features.T
        image_similarity = image_features @ image_features.T
        text_similarity = text_features @ text_features.T

        targets = F.softmax(
            image_similarity + text_similarity, dim=-1
        )

        texts_loss = cross_entropy(logits, targets, reduction='none')
        images_loss = cross_entropy(logits.T, targets.T, reduction='none')

        loss = (images_loss + texts_loss) / 2.0
        return logits, loss.mean(), text_features, image_features
