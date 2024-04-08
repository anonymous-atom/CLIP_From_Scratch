from torch import nn
from transformers import AutoTokenizer, T5EncoderModel, ViTForImageClassification, ViTFeatureExtractor


class TextEncoder(nn.Module):
    def __init__(self):
        super(TextEncoder, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-small")
        self.model = T5EncoderModel.from_pretrained("google-t5/t5-small")

    def forward(self, text):
        input_ids = self.tokenizer(text, return_tensors="pt").input_ids  # Batch size 1
        outputs = self.model(input_ids=input_ids)
        text_embeddings = outputs.last_hidden_state
        return text_embeddings


class ImageEncoder(nn.Module):
    def __init__(self):
        super(ImageEncoder, self).__init__()
        self.model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")
        self.extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224")

    def forward(self, image):
        image_embeddings = self.model.vit(image).last_hidden_state[:, 0]
        return image_embeddings
