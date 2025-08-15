import torch
import torch.nn as nn
from torchvision import models

# Load the original full-precision model
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
num_ftrs = model.fc.in_features
num_classes = 101 # Assuming 101 classes
model.fc = nn.Linear(num_ftrs, num_classes)
state_dict = torch.load('food_classifier_model.pth', map_location=torch.device('cpu'))
model.load_state_dict(state_dict)

# Set the model to evaluation mode
model.eval()

# Quantize the model
# This process reduces the size by converting 32-bit floating-point weights to 8-bit integers
model_quantized = torch.quantization.quantize_dynamic(
    model, 
    {torch.nn.Linear}, 
    dtype=torch.qint8
)

# Save the quantized model
torch.save(model_quantized.state_dict(), 'food_classifier_model_quantized.pth')

print("Model quantization complete. A new, smaller model file has been created.")