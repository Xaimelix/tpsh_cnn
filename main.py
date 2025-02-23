import torch
import torchvision.models as models
from torchvision import transforms
from torchvision.models.vgg import VGG16_Weights
from PIL import Image
import json


vgg16 = models.vgg16(weights=VGG16_Weights.DEFAULT)
vgg16.eval()

with open("imagenet_class_index.json") as f:
        class_idx = json.load(f)
idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

file_path = 'cat.jpg'
image = Image.open(file_path)
input_tensor = preprocess(image)
input_batch = input_tensor.unsqueeze(0)

with torch.no_grad():
    output = vgg16(input_batch)

probabilities = torch.nn.functional.softmax(output[0], dim=0)
_, top_class_idx = torch.topk(probabilities, 1) # Получаем только лучший класс

predicted_class_label = idx2label[top_class_idx[0].item()]
confidence = probabilities[top_class_idx[0].item()].item()

print(predicted_class_label, confidence)
