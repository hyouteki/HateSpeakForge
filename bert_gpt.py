import torch
from transformers import BertTokenizer, BertModel, GPT2Tokenizer, GPT2LMHeadModel
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from datasets import load_dataset
import re
import html
from tqdm import tqdm
from datasets import load_metric

dataset = load_dataset("social_bias_frames")

regex_patterns = {
    "RT": r"RT @[^:]+:", 
    "URL": r"https?://\S+|www\.\S+", 
    "PUNC": r"[.,?!#@]",
    "SPC_PUNC": r"[-']"
} 

trainSplitSize = 100
valSplitSize = 100
testSplitSize = 10

def preprocess_dataframe(dataframe):
    for i, post in enumerate(dataframe["post"].to_list()):
        post = html.unescape(post)
        for key, reg in regex_patterns.items(): 
            delim = ' ' if key == "SPC_PUNC" else ''
            post = re.sub(reg, delim, post) 
        post = post.strip()
        dataframe.loc[i, "post"] = post

class TextDataset(Dataset):
    def __init__(self, src_texts, tgt_texts, tokenizer, max_length=512):
        self.src_texts = src_texts
        self.tgt_texts = tgt_texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.src_texts)
    
    def __getitem__(self, idx):
        src_text = self.src_texts[idx]
        tgt_text = self.tgt_texts[idx]
        
        src_encoding = self.tokenizer(src_text, max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt")
        tgt_encoding = self.tokenizer(tgt_text, max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt")
        
        return {
            "src_input_ids": src_encoding["input_ids"].flatten(),
            "src_attention_mask": src_encoding["attention_mask"].flatten(),
            "tgt_input_ids": tgt_encoding["input_ids"].flatten(),
            "tgt_attention_mask": tgt_encoding["attention_mask"].flatten(),
        }

def get_tokenized_dataset(partition, size, tokenizer):
    dataframe = pd.DataFrame(dataset[partition] if size is None else dataset[partition].select(range(size)))
    dataframe.drop_duplicates(subset=["post", "targetStereotype"], keep="first", inplace=True)   
    dataframe.dropna(inplace=True) 
    preprocess_dataframe(dataframe)
    srcSamples, tgtSamples = [], []
    for _, sample in dataframe.iterrows():
        tgtText = sample["targetStereotype"]
        if pd.isna(tgtText) or len(tgtText) == 0:
            continue
        srcSamples.append(sample["post"].rstrip())
        tgtSamples.append(tgtText)
    return TextDataset(srcSamples, tgtSamples, tokenizer)


# Load pre-trained BERT tokenizer and model
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

# Load pre-trained GPT-2 tokenizer and model
from transformers import GPT2Config

# Load pre-trained GPT-2 configuration
gpt_config = GPT2Config.from_pretrained('gpt2', add_cross_attention=True)

# Load pre-trained GPT-2 model with cross-attention layers
gpt_model = GPT2LMHeadModel.from_pretrained('gpt2', config=gpt_config)


# Get tokenized datasets
train_dataset = get_tokenized_dataset("train", trainSplitSize, bert_tokenizer)
val_dataset = get_tokenized_dataset("validation", valSplitSize, bert_tokenizer)

# Fine-tuning parameters
batch_size = 32
num_epochs = 5
learning_rate = 5e-5

# DataLoader for batch processing
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

# Set device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Move models to device
bert_model.to(device)
gpt_model.to(device)

# Optimizer
optimizer = torch.optim.AdamW(gpt_model.parameters(), lr=learning_rate)

# Function to compute metrics
def compute_metrics(predictions, label_ids):
    print("bert f1 calculating")
    bert_f1s = load_metric("bertscore").compute(predictions=[predictions], references=[label_ids], lang="en")["f1"]
    bert_f1_avg = sum(bert_f1s)/len(bert_f1s)
    print(f"bert f1: {bert_f1_avg}")
    print("bleu calculating")
    bleu_score = load_metric("bleu").compute(predictions=[predictions], references=[label_ids], max_order=2)["bleu"]
    print(f"bleu score: {bleu_score}")
    print("rouge_l calculating")
    rouge_l = load_metric("rouge").compute(predictions=[predictions], references=[label_ids])["rougeL"]
    print(f"rouge l: {rouge_l}")
    return {
        "bleu": bleu_score,
        "rougeL": rouge_l,
        "bertscore": bert_f1_avg
    }

trainLosses = []
valLosses = []

# Training loop
for epoch in range(num_epochs):
    gpt_model.train()
    total_loss = 0
    for batch in tqdm(train_dataloader):
        src_input_ids = batch["src_input_ids"].to(device)
        src_attention_mask = batch["src_attention_mask"].to(device)
        tgt_input_ids = batch["tgt_input_ids"].to(device)
        tgt_attention_mask = batch["tgt_attention_mask"].to(device)

        # Encode source text with BERT
        with torch.no_grad():
            bert_output = bert_model(input_ids=src_input_ids, attention_mask=src_attention_mask).last_hidden_state
        
        # Decode with GPT-2
        outputs = gpt_model(input_ids=tgt_input_ids, attention_mask=tgt_attention_mask, encoder_hidden_states=bert_output, labels=tgt_input_ids)
        loss = outputs.loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()

    trainLosses.append(total_loss/len(train_dataloader))
    print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_dataloader)}")

    # Evaluation
    gpt_model.eval()
    total_val_loss = 0
    all_predictions = []
    all_label_ids = []
    with torch.no_grad():
        for batch in tqdm(val_dataloader):
            src_input_ids = batch["src_input_ids"].to(device)
            src_attention_mask = batch["src_attention_mask"].to(device)
            tgt_input_ids = batch["tgt_input_ids"].to(device)
            tgt_attention_mask = batch["tgt_attention_mask"].to(device)

            bert_output = bert_model(input_ids=src_input_ids, attention_mask=src_attention_mask).last_hidden_state
            
            outputs = gpt_model(input_ids=tgt_input_ids, attention_mask=tgt_attention_mask, encoder_hidden_states=bert_output, labels=tgt_input_ids)
            total_val_loss += outputs.loss.item()

            # Generate predictions for metrics computation
            generated = gpt_model.generate(input_ids=src_input_ids, attention_mask=src_attention_mask, max_length=720, pad_token_id=gpt_model.config.eos_token_id)
            all_predictions.extend(generated)
            all_label_ids.extend(tgt_input_ids.tolist())

    valLosses.append(total_val_loss/len(val_dataloader))
    print(f"Epoch {epoch+1}, Validation Loss: {total_val_loss/len(val_dataloader)}")

print(trainLosses)
print(valLosses)

torch.save(bert_model, "bg_bert.pt")
torch.save(gpt_model, "bg_gpt.pt")