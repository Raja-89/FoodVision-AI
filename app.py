import torch
import torch.nn as nn
from torchvision import models, transforms
from flask import Flask, request, jsonify, render_template

from flask_cors import CORS
from PIL import Image
import io
import base64
import os
import requests
import numpy as np

app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    return render_template('index.html')

# Define your food classes from the dataset
FOOD_CLASSES = [
    "apple_pie", "baby_back_ribs", "baklava", "beef_carpaccio", "beef_tartare",
    "beet_salad", "beignets", "bibimbap", "bread_pudding", "breakfast_burrito",
    "bruschetta", "caesar_salad", "cannoli", "caprese_salad", "carrot_cake",
    "ceviche", "cheesecake", "cheese_plate", "chicken_curry", "chicken_quesadilla",
    "chicken_wings", "chocolate_cake", "chocolate_mousse", "churros", "clam_chowder",
    "club_sandwich", "crab_cakes", "creme_brulee", "croque_madame", "cup_cakes",
    "deviled_eggs", "donuts", "dumplings", "edamame", "eggs_benedict",
    "escargots", "falafel", "filet_mignon", "fish_and_chips", "foie_gras",
    "french_fries", "french_onion_soup", "french_toast", "fried_calamari", "fried_rice",
    "frozen_yogurt", "garlic_bread", "gnocchi", "greek_salad", "grilled_cheese_sandwich",
    "grilled_salmon", "guacamole", "gyoza", "hamburger", "hot_and_sour_soup",
    "hot_dog", "huevos_rancheros", "hummus", "ice_cream", "lasagna",
    "lobster_bisque", "lobster_roll_sandwich", "macaroni_and_cheese", "macarons", "miso_soup",
    "mussels", "nachos", "omelette", "onion_rings", "oysters",
    "pad_thai", "paella", "pancakes", "panna_cotta", "peking_duck",
    "pho", "pizza", "pork_chop", "poutine", "prime_rib",
    "pulled_pork_sandwich", "ramen", "ravioli", "red_velvet_cake", "risotto",
    "samosa", "sashimi", "scallops", "seaweed_salad", "shrimp_and_grits",
    "spaghetti_bolognese", "spaghetti_carbonara", "spring_rolls", "steak", "strawberry_shortcake",
    "sushi", "tacos", "takoyaki", "tiramisu", "tuna_tartare",
    "waffles",
]

# --- Model Loading Configuration ---
# Your new public download URL for the quantized model
MODEL_URL = "https://drive.google.com/file/d/15oT_a-hPHFHCMUwseP4LCSBUgf9iQC7t/view?usp=sharing"
MODEL_PATH = "food_classifier_model_quantized.pth"

# Re-instantiate the same model architecture used for training
# Re-instantiate the same model architecture used for training
def load_model():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    num_ftrs = model.fc.in_features
    num_classes = len(FOOD_CLASSES)
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    # Apply dynamic quantization to the model architecture BEFORE loading the weights
    model_quantized = torch.quantization.quantize_dynamic(
        model, 
        {torch.nn.Linear}, 
        dtype=torch.qint8
    )

    # Load the saved state_dict into the quantized model
    state_dict = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
    model_quantized.load_state_dict(state_dict)

    # Set the model to evaluation mode
    model_quantized.eval()

    return model_quantized

# Load the model once when the application starts
try:
    if not os.path.exists(MODEL_PATH):
        print("Model file not found. Downloading from URL...")
        with requests.get(MODEL_URL, stream=True) as r:
            r.raise_for_status()
            with open(MODEL_PATH, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print("Model downloaded successfully.")
        
    model = load_model()
    print("PyTorch model loaded successfully.")
except Exception as e:
    print(f"Error loading the PyTorch model: {e}")
    model = None

# --- Preprocessing Transformations ---
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- Prediction Endpoint ---
@app.route('/classify', methods=['POST'])
def classify_image():
    if model is None:
        return jsonify({'error': 'Model could not be loaded.'}), 500

    try:
        image_data = request.json['image']
        image_data = image_data.split(',')[1]
        image_bytes = base64.b64decode(image_data)
        
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        image_tensor = preprocess(image).unsqueeze(0)
        
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
        
        top_probabilities, top_indices = torch.topk(probabilities, 5)
        
        results = []
        for i in range(5):
            results.append({
                'class': FOOD_CLASSES[top_indices[i].item()],
                'confidence': top_probabilities[i].item()
            })
        
        return jsonify({'predictions': results})
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)