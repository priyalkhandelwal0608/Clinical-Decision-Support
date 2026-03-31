# Clinical Symptom Classifier

##  Overview
The **Clinical Symptom Classifier** is an AI-powered tool that predicts potential diseases based on symptoms described by the user. It uses a **BERT-based model** to analyze textual symptom inputs and provides both:

- **Top disease predictions** with confidence scores
- **Word-level attention** highlighting key symptoms influencing the prediction

The project also features an interactive **Gradio web interface** for quick testing without writing any code.

---

##  Features
- Preprocess and encode symptoms using **BERT tokenizer**
- Predict diseases using a **fine-tuned BERT classifier**
- Display **attention scores** for explainability
- Interactive web UI powered by **Gradio**
- Easy to extend for new diseases or symptoms

---

## Installation and run
- pip install -r requirements.txt
- python run app.py
