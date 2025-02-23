import io
import torch
import torchvision.models as models
from torchvision import transforms
from torchvision.models.vgg import VGG16_Weights
from PIL import Image
import json
from flask import Flask, render_template, request, jsonify


app = Flask(__name__)

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

def predict_image(image_bytes):
    try:
        image = Image.open(io.BytesIO(image_bytes))
        input_tensor = preprocess(image)
        input_batch = input_tensor.unsqueeze(0)

        with torch.no_grad():
            output = vgg16(input_batch)

        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        _, top_class_idx = torch.topk(probabilities, 1) # Здесь мы получаем индекс класса с наибольшей вероятностью

        predicted_class_label = idx2label[top_class_idx[0].item()]
        confidence = probabilities[top_class_idx[0].item()].item()

        return predicted_class_label, confidence

    except Exception as e:
        return None, str(e)


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'Нет изображения в запросе'}), 400

    file = request.files['image']

    if file.filename == '':
        return jsonify({'error': 'Нет выбранного файла'}), 400

    try:
        image_bytes = file.read()
        predicted_label, confidence = predict_image(image_bytes)

        if predicted_label:
            return jsonify({'prediction': predicted_label, 'confidence': f'{confidence:.4f}'})
        else:
            return jsonify({'error': f'Ошибка при обработке изображения: {confidence}'}), 500
    except Exception as e:
        return jsonify({'error': f'Ошибка: {str(e)}'}), 500


app.run(debug=True) 