import torch
from transformers import BertTokenizer
from app.model import SentimentClassifier

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_prediction(text):
    model = SentimentClassifier()
    model.to(device)
    model.eval()

    inputs = tokenizer.encode_plus(
        text,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    ids = inputs['input_ids'].to(device)
    mask = inputs['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(ids, mask)
        _, prediction = torch.max(outputs, dim=1)
    
    labels = ["Negative", "Neutral", "Positive"]
    return labels[prediction.item()]
