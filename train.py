import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AvgMeter:
    def __init__(self, name="Metric"):
        self.name = name
        self.reset()

    def reset(self):
        self.avg, self.sum, self.count = [0] * 3

    def update(self, val, count=1):
        self.count += count
        self.sum += val * count
        self.avg = self.sum / self.count

    def __repr__(self):
        text = f"{self.name}: {self.avg:.4f}"
        return text

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]

class CLIPDataset(Dataset):
    def __init__(self, texts, images, transform=None):
        self.texts = texts
        self.images = images
        self.transform = transform

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        image = Image.open(self.images[idx])
        if self.transform:
            image = self.transform(image)
        else:
            to_tensor = T.ToTensor()
            image = to_tensor(image)
        return text, image


# # Create sample array of texts and images
# texts = ["A photo of a dog", "A photo of a bird"]
# images = ["images/dog.jpeg", "images/bird.jpg"]
#
# # Create dataset and dataloader
# dataset = CLIPDataset(texts, images)
# dataloader = DataLoader(dataset, batch_size=1)


def train_step(model, data, optimizer):
    # Move model to a device

    model.train()
    text, image = data

    optimizer.zero_grad()
    logits, loss, _, _ = model(image, text)
    loss.backward()
    optimizer.step()
    return loss.item()


def train(model, dataloader, optimizer, num_epochs=10):
    loss_meter = AvgMeter()
    for epoch in range(num_epochs):
        for i, data in enumerate(dataloader):
            loss = train_step(model, data, optimizer)
            loss_meter.update(loss, count=1)
            loss = loss_meter.avg
            print(f"Epoch {epoch}, Batch {i}, Loss: {loss}")
        print(f"Epoch {epoch}, Loss: {loss_meter.avg}")
