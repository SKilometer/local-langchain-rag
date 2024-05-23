import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import BertTokenizer, BertModel
from transformers import AutoTokenizer, AutoModelForMaskedLM


class RankerDataset(Dataset):
    def __init__(self, pairs, tokenizer):
        self.pairs = pairs
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):

        row = self.pairs.iloc[idx]
        query = row['Rewritten Query']
        label = row['Score']
        question_original = row['Original Question']

        combined_text = f"{query}\n <------------------->\n {question_original}"
        combined_enc = self.tokenizer(combined_text, return_tensors='pt', padding='max_length', max_length=512,
                                      truncation=True)

        return {
            'input_ids': combined_enc['input_ids'].squeeze(0),
            'attention_mask': combined_enc['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.float)
        }


class Ranker(nn.Module):
    def __init__(self, bert_model):
        super(Ranker, self).__init__()
        self.bert = bert_model
        self.fc = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, 512),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output  # 使用[CLS]标记的表示
        score = self.fc(pooled_output)
        return score


# tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
# bert_model = AutoModelForMaskedLM.from_pretrained("bert-base-chinese")

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')  # 维度 768

# 调用加载函数
training_data = pd.read_csv('./training_data/liver_training_data.csv')
print(training_data.shape)
dataset = RankerDataset(training_data, tokenizer)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
ranker = Ranker(bert_model).to(device)
optimizer = optim.Adam(ranker.parameters(), lr=0.001)
mse_loss = nn.MSELoss()
num_epochs = 100

# train
for epoch in tqdm(range(num_epochs)):
    ranker.train()
    epoch_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        scores = ranker(input_ids, attention_mask)
        loss = mse_loss(scores.squeeze(), labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    print(f"Epoch [{epoch + 1}/{num_epochs}],Loss:{epoch_loss / len(train_loader):.4f}")

# test
ranker.eval()
test_loss = 0
with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        scores = ranker(input_ids, attention_mask)
        loss = mse_loss(scores.squeeze(), labels)
        test_loss += loss.item()

    print(f"Test Loss: {test_loss / len(test_loader):.4f}")

# save
torch.save(ranker.state_dict(), "pairwise_ranker.pth")
print("Model saved as 'pairwise_ranker.pth'")
