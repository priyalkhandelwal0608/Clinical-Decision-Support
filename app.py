import os
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel, logging
import pandas as pd
import gradio as gr
from sklearn.preprocessing import LabelEncoder

# ---------------------------
# 0. Suppress Warnings
# ---------------------------
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"
warnings.filterwarnings("ignore")
logging.set_verbosity_error()  # suppress HF warnings

# ---------------------------
# 1. Load Dataset & Labels
# ---------------------------
df = pd.read_csv("data.csv").dropna()
texts = df['text'].tolist()
labels = df['label'].tolist()

le = LabelEncoder()
label_ids = le.fit_transform(labels)
classes = le.classes_

# ---------------------------
# 2. Device & Tokenizer
# ---------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TOKENIZER = BertTokenizer.from_pretrained('bert-base-cased', use_auth_token=False)

# ---------------------------
# 3. Dummy Classifier for Immediate Testing
# ---------------------------
# This simulates predictions so Gradio UI works instantly.
# Later, you can replace it with your trained model.
class DummyClassifier:
    def __init__(self, classes):
        self.classes = classes

    def predict(self, text):
        import random
        # Random probabilities
        probs = [random.random() for _ in self.classes]
        total = sum(probs)
        probs = [p/total for p in probs]
        confidences = {self.classes[i]: probs[i] for i in range(len(self.classes))}

        # Fake attention explanation
        tokens = text.split()
        word_importance = ""
        for t in tokens:
            score = random.uniform(0, 1)
            word_importance += f"{t} ({score:.2f})  "
        
        return confidences, f"Key Symptoms Detected: {word_importance}"

# Initialize dummy model
model = DummyClassifier(classes)

# ---------------------------
# 4. Prediction Function
# ---------------------------
def predict_symptoms(text):
    return model.predict(text)

# ---------------------------
# 5. Gradio Interface
# ---------------------------
desc = """
## 🩺 Advanced Symptom-to-Disease Classifier
*Type your symptoms below to see the predicted disease and key symptoms detected.*
"""

interface = gr.Interface(
    fn=predict_symptoms,
    inputs=gr.Textbox(lines=3, placeholder="Describe your symptoms..."),
    outputs=[
        gr.Label(num_top_classes=3, label="Top Diagnoses"),
        gr.Markdown(label="Explainability")
    ],
    title="Clinical Symptom Classifier",
    description=desc,
    examples=[
        ["I have high fever and dry cough."],
        ["My skin is itchy with red scaly patches."],
        ["I have sneezing and watery eyes after going outside."]
    ]
)

interface.launch(share=True, debug=False)