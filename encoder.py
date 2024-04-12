from torch import nn
from transformers import AutoTokenizer, T5EncoderModel,ViTImageProcessor, ViTModel



class TextEncoder(nn.Module):
    def __init__(self):
        super(TextEncoder, self).__init__()
        print("Loading Text Encoder....")
        self.tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-small")
        self.model = T5EncoderModel.from_pretrained("google-t5/t5-small")

    def forward(self, text):
        # Encode the text, add padding and truncation
        encoding = self.tokenizer(text, return_tensors="pt", padding='longest', truncation=True, max_length=512)
        input_ids = encoding.input_ids
        outputs = self.model(input_ids=input_ids)
        text_embeddings = outputs.last_hidden_state
        return text_embeddings


class ImageEncoder(nn.Module):
    def __init__(self):
        super(ImageEncoder, self).__init__()
        print("Loading Image Encoder....")
        self.processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
        self.model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k', )

    def forward(self, image):
        inputs = self.processor(images=image, return_tensors="pt")
        outputs = self.model(**inputs)
        image_embeddings = outputs.last_hidden_state
        return image_embeddings
