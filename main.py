from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.preprocessing import LabelEncoder
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os  # جديد

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
