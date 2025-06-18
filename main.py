from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.preprocessing import LabelEncoder
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import requests
import zipfile

# ========== Download and Extract Model ==========

def download_and_extract_model():
    model_dir = "bert_emotion_model"
    if not os.path.exists(model_dir):
        print("🔽 Downloading model zip file...")
        url = "https://drive.google.com/uc?export=download&id=1edAuAq4w5lOPUB5wfYrDcp9mgE_fSlj7"
        zip_path = "bert_emotion_model.zip"
        
        response = requests.get(url)
        with open(zip_path, "wb") as f:
            f.write(response.content)
        
        print("📦 Extracting model files...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(model_dir)
        
        os.remove(zip_path)
        print("✅ Model is ready.")

download_and_extract_model()

# ========== Load Model and Data ==========

model_path = "./bert_emotion_model"
responses_csv_path = "./emotional_chatbot_responses.csv"

tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

responses_df = pd.read_csv(responses_csv_path, encoding='latin1')

emotion_labels = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']
le = LabelEncoder()
le.fit(emotion_labels)

# ========== Prediction Function ==========

def predict_emotion(text):
    with torch.no_grad():
        encoding = tokenizer(text, return_tensors='pt', truncation=True, padding='max_length', max_length=128)
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_label = torch.argmax(probs, dim=1).item()
        return le.inverse_transform([predicted_label])[0]

# ========== FastAPI Setup ==========

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

class Message(BaseModel):
    text: str

@app.post("/emotion-response/")
def get_emotion_response(message: Message):
    detected_emotion = predict_emotion(message.text)
    matched = responses_df[responses_df['emotion'] == detected_emotion]

    if not matched.empty:
        response_text = matched.iloc[0]['response']
    else:
        response_text = "I'm here for you. Please tell me more."

    return {
        "detected_emotion": detected_emotion,
        "response": response_text
    }

# ========== Run the Server ==========

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))
