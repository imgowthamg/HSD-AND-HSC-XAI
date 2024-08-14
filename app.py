'''import gradio as gr
import pandas as pd
import numpy as np
import re
import h5py
from tensorflow.keras.models import model_from_json, Sequential
from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, Bidirectional, LSTM, Dropout, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.stem import WordNetLemmatizer
from lime.lime_text import LimeTextExplainer
import matplotlib.pyplot as plt

# Function to load model and remove unrecognized argument
def load_custom_model(h5_path):
    with h5py.File(h5_path, 'r') as f:
        model_config = f.attrs.get('model_config')
        if model_config:
            model_config = model_config.replace('"time_major": false,', '')  # Remove the unrecognized argument
            custom_objects = {
                'Sequential': Sequential,
                'Embedding': Embedding,
                'Conv1D': Conv1D,
                'MaxPooling1D': MaxPooling1D,
                'Bidirectional': Bidirectional,
                'LSTM': LSTM,
                'Dropout': Dropout,
                'Dense': Dense
            }
            model = model_from_json(model_config, custom_objects=custom_objects)
            model.load_weights(h5_path)
            return model
        else:
            raise ValueError('No model configuration found in the .h5 file.')

# Load the classification model
hsc_model = load_custom_model('hsc_model.h5')

# Preprocessing functions
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = ' '.join(lemmatizer.lemmatize(word) for word in text.split())
    text = re.sub(r'@\w+', '', text)  # Remove usernames
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'[^A-Za-z0-9\s]', '', text)  # Remove special characters
    text = text.lower().strip()  # Lowercase and strip whitespaces
    return text

# Load and fit tokenizer
hsc = pd.read_csv("Classification.csv")
tokenizer = Tokenizer(num_words=10000)  # Use the same max_words as in your training
tokenizer.fit_on_texts(hsc['tweet'].astype(str))

def model_predict_proba(texts):
    sequences = tokenizer.texts_to_sequences(texts)
    padded_sequences = pad_sequences(sequences, maxlen=150)
    return hsc_model.predict(padded_sequences)

def get_lime_explanation(text, num_features=10):
    class_names = ["Anti-State", "Anti-Religion", "Offensive", "Sexism", "Racism"]
    explainer = LimeTextExplainer(class_names=class_names)
    exp = explainer.explain_instance(
        text, 
        model_predict_proba, 
        num_features=num_features, 
        top_labels=1
    )
    return exp

def predict_hate_speech(text):
    preprocessed_text = preprocess_text(text)
    sequence = tokenizer.texts_to_sequences([preprocessed_text])
    padded_sequence = pad_sequences(sequence, maxlen=150)  # Use the same max_len as in your training
    
    # Classify hate speech
    probabilities = hsc_model.predict(padded_sequence)[0]
    category_index = np.argmax(probabilities)
    labels = ["Anti-State", "Anti-Religion", "Offensive", "Sexism", "Racism"]
    category_label = labels[category_index]
    
    # Get LIME explanation
    lime_exp = get_lime_explanation(preprocessed_text)
    explanation = lime_exp.as_list(label=category_index)
    
    # Create explanation string
    exp_string = f"Category: {category_label}\n\nLIME Explanation:\n"
    for feature, importance in explanation:
        exp_string += f"{feature}: {importance:.4f}\n"
    
    # Create and save LIME plot
    plt.figure(figsize=(10, 6))
    lime_exp.as_pyplot_figure(label=category_index)
    plt.title(f"LIME Explanation for class '{category_label}'")
    plt.tight_layout()
    plt.savefig('lime_explanation.png')
    plt.close()
    
    return exp_string, 'lime_explanation.png'

# Create Gradio interface
interface = gr.Interface(
    fn=predict_hate_speech, 
    inputs="text", 
    outputs=[gr.Textbox(label="Prediction and Explanation"), gr.Image(label="LIME Visualization")],
    title="Hate Speech Classification with LIME Explanation",
    description="Enter a text to classify it and get a LIME explanation."
)
interface.launch()




import gradio as gr
import torch
from transformers import BertModel, BertTokenizer
import torch.nn as nn
import re

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pre-trained BERT model and tokenizer
bert_model = BertModel.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

class HateSpeechClassifier(nn.Module):
    def __init__(self):
        super(HateSpeechClassifier, self).__init__()
        self.bert = bert_model
        self.dropout = nn.Dropout(0.1)
        self.linear1 = nn.Linear(768, 256)
        self.linear2 = nn.Linear(256, 2)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        pooled_output = outputs[0][:, 0, :]  # Use [CLS] token
        x = self.dropout(pooled_output)
        x = torch.relu(self.linear1(x))
        x = self.linear2(x)
        return x

# Load the trained model
model = HateSpeechClassifier()
model.load_state_dict(torch.load('hate_speech_model.pt', map_location=device))
model.to(device)
model.eval()

def clean_text(text):
    text = re.sub(r"@[A-Za-z0-9_-]+", 'USR', text)
    text = re.sub(r"http\S+", 'URL', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text.lower()

def predict_hate_speech(text):
    cleaned_text = clean_text(text)
    encoded_text = tokenizer.encode_plus(
        cleaned_text,
        max_length=60,
        truncation=True,
        padding='max_length',
        return_tensors='pt'
    )
    input_ids = encoded_text['input_ids'].to(device)
    attention_mask = encoded_text['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        probabilities = torch.softmax(outputs, dim=1)
        prediction = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][prediction].item()

    if prediction == 1:
        result = f"This text is classified as hate speech with {confidence:.2%} confidence."
    else:
        result = f"This text is not classified as hate speech with {confidence:.2%} confidence."

    return result

# Create Gradio interface
interface = gr.Interface(
    fn=predict_hate_speech,
    inputs=gr.Textbox(lines=2, placeholder="Enter text here..."),
    outputs=gr.Textbox(),
    title="Hate Speech Detection",
    description="Enter a text to detect if it's hate speech or not."
)

# Launch the interface
interface.launch()'''



import gradio as gr
import torch
import tensorflow as tf
import numpy as np
import re
import h5py
import pandas as pd
import html
from transformers import BertModel, BertTokenizer as BertTokenizerTransformers
import torch.nn as nn
from tensorflow.keras.models import model_from_json, Sequential
from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, Bidirectional, LSTM, Dropout, Dense
from tensorflow.keras.preprocessing.text import Tokenizer as KerasTokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.stem import WordNetLemmatizer
from lime.lime_text import LimeTextExplainer
import matplotlib.pyplot as plt

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pre-trained BERT model and tokenizer
bert_model = BertModel.from_pretrained('bert-base-uncased')
bert_tokenizer = BertTokenizerTransformers.from_pretrained('bert-base-uncased')

# BERT model for initial hate speech detection
class HateSpeechClassifier(nn.Module):
    def __init__(self):
        super(HateSpeechClassifier, self).__init__()
        self.bert = bert_model
        self.dropout = nn.Dropout(0.1)
        self.linear1 = nn.Linear(768, 256)
        self.linear2 = nn.Linear(256, 2)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        pooled_output = outputs[0][:, 0, :]  # Use [CLS] token
        x = self.dropout(pooled_output)
        x = torch.relu(self.linear1(x))
        x = self.linear2(x)
        return x

# Load the trained BERT model
bert_classifier = HateSpeechClassifier()
bert_classifier.load_state_dict(torch.load('hate_speech_model.pt', map_location=device))
bert_classifier.to(device)
bert_classifier.eval()

# Function to load custom model
def load_custom_model(h5_path):
    with h5py.File(h5_path, 'r') as f:
        model_config = f.attrs.get('model_config')
        if model_config:
            model_config = model_config.replace('"time_major": false,', '')  # Remove the unrecognized argument
            custom_objects = {
                'Sequential': Sequential,
                'Embedding': Embedding,
                'Conv1D': Conv1D,
                'MaxPooling1D': MaxPooling1D,
                'Bidirectional': Bidirectional,
                'LSTM': LSTM,
                'Dropout': Dropout,
                'Dense': Dense
            }
            model = model_from_json(model_config, custom_objects=custom_objects)
            model.load_weights(h5_path)
            return model
        else:
            raise ValueError('No model configuration found in the .h5 file.')

# Load the custom classification model
hsc_model = load_custom_model('hsc_model.h5')

# Preprocessing functions
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = re.sub(r"@[A-Za-z0-9_-]+", 'USR', text)
    text = re.sub(r"http\S+", 'URL', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text.lower()

def preprocess_text(text):
    text = ' '.join(lemmatizer.lemmatize(word) for word in text.split())
    text = re.sub(r'@\w+', '', text)  # Remove usernames
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'[^A-Za-z0-9\s]', '', text)  # Remove special characters
    text = text.lower().strip()  # Lowercase and strip whitespaces
    return text

# Load and fit tokenizer for custom model
hsc = pd.read_csv("Classification.csv")
keras_tokenizer = KerasTokenizer(num_words=10000)  # Use the same max_words as in your training
keras_tokenizer.fit_on_texts(hsc['tweet'].astype(str))

def model_predict_proba(texts):
    sequences = keras_tokenizer.texts_to_sequences(texts)
    padded_sequences = pad_sequences(sequences, maxlen=150)
    return hsc_model.predict(padded_sequences)

def get_lime_explanation(text, num_features=10):
    class_names = ["Anti-State", "Anti-Religion", "Offensive", "Sexism", "Racism"]
    explainer = LimeTextExplainer(class_names=class_names)
    exp = explainer.explain_instance(
        text, 
        model_predict_proba, 
        num_features=num_features, 
        top_labels=1
    )
    return exp

def detect_hate_speech(text):
    cleaned_text = clean_text(text)
    encoded_text = bert_tokenizer.encode_plus(
        cleaned_text,
        max_length=60,
        truncation=True,
        padding='max_length',
        return_tensors='pt'
    )
    input_ids = encoded_text['input_ids'].to(device)
    attention_mask = encoded_text['attention_mask'].to(device)

    with torch.no_grad():
        outputs = bert_classifier(input_ids, attention_mask)
        probabilities = torch.softmax(outputs, dim=1)
        prediction = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][prediction].item()

    return prediction == 1, confidence

def classify_and_explain(text):
    preprocessed_text = preprocess_text(text)
    sequence = keras_tokenizer.texts_to_sequences([preprocessed_text])
    padded_sequence = pad_sequences(sequence, maxlen=150)
    
    # Classify hate speech
    probabilities = hsc_model.predict(padded_sequence)[0]
    category_index = np.argmax(probabilities)
    labels = ["Anti-State", "Anti-Religion", "Offensive", "Sexism", "Racism"]
    category_label = labels[category_index]
    
    # Get LIME explanation
    lime_exp = get_lime_explanation(preprocessed_text)
    explanation = lime_exp.as_list(label=category_index)
    
    # Create explanation string with highlighted text
    exp_string = f"<h3>Category: {category_label}</h3>"
    exp_string += "<h4>LIME Explanation:</h4>"
    exp_string += "<ul>"
    for feature, importance in explanation:
        color = "red" if importance < 0 else "green"
        exp_string += f'<li><span style="color: {color};">{html.escape(feature)}</span>: {importance:.4f}</li>'
    exp_string += "</ul>"
    
    # Get highlighted text
    highlighted_text = lime_exp.as_html(text=True)
    
    # Create and save LIME plot
    plt.figure(figsize=(10, 6))
    lime_exp.as_pyplot_figure(label=category_index)
    plt.title(f"LIME Explanation for class '{category_label}'")
    plt.tight_layout()
    plt.savefig('lime_explanation.png')
    plt.close()
    
    return category_label, exp_string, highlighted_text, 'lime_explanation.png'

def integrated_hate_speech_analysis(text):
    is_hate_speech, confidence = detect_hate_speech(text)
    
    if is_hate_speech:
        category, explanation, highlighted_text, lime_plot = classify_and_explain(text)
        result = f"<h2>Hate Speech Detected</h2>"
        result += f"<p>Confidence: {confidence:.2%}</p>"
        result += f"<p>Specific category: {category}</p>"
        result += f"<h3>Explanation:</h3>{explanation}"
        result += f"<h3>Highlighted Text:</h3>{highlighted_text}"
        return result, lime_plot
    else:
        return f"<h2>Not Hate Speech</h2><p>This text is not classified as hate speech with {confidence:.2%} confidence.</p>", None

# Create Gradio interface
interface = gr.Interface(
    fn=integrated_hate_speech_analysis, 
    inputs=gr.Textbox(label="Enter text for analysis", lines=5),
    outputs=[
        gr.HTML(label="Analysis Result"),
        gr.Image(label="LIME Visualization")
    ],
    title="Integrated Hate Speech Analysis",
    description="Enter a text to detect hate speech. If detected, it will be classified and explained.",
    theme="default"
)

interface.launch()