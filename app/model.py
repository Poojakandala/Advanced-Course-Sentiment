import torch.nn as nn
from transformers import BertModel

class SentimentClassifier(nn.Module):
    def __init__(self, n_classes=3):
        super(SentimentClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.drop = nn.Dropout(p=0.3)
        # Bi-LSTM to capture context in both directions
        self.lstm = nn.LSTM(768, 128, bidirectional=True, batch_first=True)
        self.out = nn.Linear(128 * 2, n_classes)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        _, (h_n, _) = self.lstm(outputs.last_hidden_state)
        # Concatenate final hidden states from both directions
        hidden = nn.functional.dropout(torch.cat((h_n[-2,:,:], h_n[-1,:,:]), dim=1), p=0.3)
        return self.out(hidden)
