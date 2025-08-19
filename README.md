# 🍕 FoodVision AI

### An end-to-end food classification application built with PyTorch, Flask, and a modern web interface.

Welcome to FoodVision AI, a project that demonstrates the entire machine learning lifecycle from model training and optimization to global deployment. This application can accurately classify 101 different food categories from any uploaded image.

## ✨ Features

  * **101 Food Categories**: Classifies a wide variety of dishes from the challenging Food-101 dataset.
  * **Optimized for Mobile**: A lightweight frontend with client-side image resizing for a seamless experience on any device.
  * **High-Performance API**: A Flask backend that serves a quantized PyTorch model, ensuring fast and efficient predictions.
  * **Robust Deployment**: Configured for stable deployment on cloud platforms like Render, handling large model files effectively.


## 🛠️ Tech Stack

  * **Backend**: Python, **Flask**, **Gunicorn**
  * **Machine Learning**: **PyTorch**, **torchvision**
  * **Model Optimization**: **Quantization** for a 4x size reduction
  * **Frontend**: HTML, CSS, JavaScript
  * **Deployment**: **Render**


## 🚀 Installation & Running Locally

### Prerequisites

  * Python 3.8+
  * `pip` and `git`

### Step 1: Clone the Repository

```bash
git clone https://github.com/Raja-89/FoodVision-AI.git
cd FoodVision-AI
```

### Step 2: Set up the Backend

1.  **Create a Virtual Environment**:

    ```bash
    python -m venv venv
    source venv/bin/activate  # On macOS/Linux
    .\venv\Scripts\activate   # On Windows
    ```

2.  **Install Dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

3.  **Get the Model**: Our model is too large for a standard Git repository. Run the `app.py` file, and it will automatically download the necessary model file from a public URL.

4.  **Run the Flask App**:

    ```bash
    python app.py
    ```

The application will be available at `http://127.0.0.1:5000`.


## 📂 Project Structure

```
.
├── app.py                      # Flask backend, API logic, and model handling
├── requirements.txt            # Python dependencies
├── Procfile                    # Render/Heroku deployment command
├── templates/
│   └── index.html              # Frontend web interface (HTML/CSS/JS)
├── food_classifier_model.pth   # The original PyTorch model
├── food_classifier_model_quantized.pth # The optimized, smaller model
├── .gitignore                  # Git ignore file
└── README.md                   # This file!
```

-----

##  Acknowledgements

A big thank you to **Rishabh Ranjan Singh** for his invaluable help and contributions to this project!
