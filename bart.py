import re
import html
import torch
import evaluate
import pandas as pd
import transformers
from torch.utils.data import Dataset
from datasets import load_dataset, load_metric
from transformers import (AutoTokenizer, AutoModelForSeq2SeqLM, 
                          Seq2SeqTrainingArguments, Seq2SeqTrainer)

modelName = "facebook/bart-base"
tokenizer = AutoTokenizer.from_pretrained(modelName)
model = AutoModelForSeq2SeqLM.from_pretrained(modelName)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.cuda.empty_cache()
model.to(device)
collator = transformers.DataCollatorForSeq2Seq(tokenizer, model=model)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)

maxSrcLen = 512
maxTgtLen = 128
batchSize = 32

trainSplitSize = 40000
valSplitSize = 8000
testSplitSize = 2000

dataset = load_dataset("social_bias_frames")

class MyDataset(Dataset):
    def __init__(self, inputIds, attentionMask, labels):
        self.inputIds = inputIds
        self.attentionMask = attentionMask
        self.labels = labels
        
    def __len__(self):
        return len(self.inputIds)

    def __getitem__(self, idx):
        return {"input_ids": self.inputIds[idx],
                "attention_mask": self.attentionMask[idx],
                "labels": self.labels[idx]}
    
regexs = {
    "RT": r"RT @[^:]+:", 
    "URL": r"https?://\S+|www\.\S+", 
    "PUNC": r"[.,?!#@]",
    "SPC_PUNC": r"[-']"
} 
def preprocessDataframe(dataframe):
    for i, post in enumerate(dataframe["post"].to_list()):
        post = html.unescape(post)
        for key, reg in regexs.items(): 
            delim = ' ' if key == "SPC_PUNC" else ''
            post = re.sub(reg, delim, post) 
        post = post.strip()
        dataframe.loc[i, "post"] = post

def getTokenizedDataset(partition, size, tokenizer):
    dataframe = pd.DataFrame(dataset[partition] if size is None else dataset[partition].select(range(size)))
    dataframe.drop_duplicates(subset=["post", "targetStereotype"], keep="first", inplace=True)   
    dataframe.dropna(inplace=True) 
    preprocessDataframe(dataframe)
    srcSamples, tgtSamples = [], []
    for _, sample in dataframe.iterrows():
        tgtText = sample["targetStereotype"]
        if pd.isna(tgtText) or len(tgtText) == 0:
            continue
        srcSamples.append(sample["post"].rstrip())
        tgtSamples.append(tgtText)
    tokenizerParams = {"padding": "max_length", "padding": True, "truncation": True}   
    tokenizedSrcSamples = tokenizer(srcSamples, max_length=maxSrcLen, **tokenizerParams)
    with tokenizer.as_target_tokenizer():
        tokenizedTgtSamples = tokenizer(tgtSamples,  max_length=maxTgtLen, **tokenizerParams)
    return MyDataset(tokenizedSrcSamples["input_ids"], 
                     tokenizedSrcSamples["attention_mask"], 
                     tokenizedTgtSamples["input_ids"])

tokenizedTrainDataset = getTokenizedDataset("train", trainSplitSize, tokenizer)
tokenizedValidationDataset = getTokenizedDataset("validation", valSplitSize, tokenizer)

def computeMetrics(p):
    predictedSamples = tokenizer.batch_decode(p.predictions, skip_special_tokens=True)
    actualSamples = tokenizer.batch_decode(p.label_ids, skip_special_tokens=True)
    bertF1s = evaluate.load("bertscore").compute(predictions=predictedSamples, references=actualSamples, lang="en")["f1"]
    return {"bleu": evaluate.load("bleu").compute(predictions=predictedSamples, references=actualSamples, max_order=2)["bleu"],
            "rougeL": evaluate.load("rouge").compute(predictions=predictedSamples, references=actualSamples)["rougeL"],
            "bertscore": sum(bertF1s)/len(bertF1s)}

args = Seq2SeqTrainingArguments(
    "bart_output",
    evaluation_strategy="epoch",
    logging_strategy="epoch",
    per_device_train_batch_size=batchSize,
    per_device_eval_batch_size=batchSize,
    gradient_accumulation_steps=1,
    weight_decay=0.01,
    logging_dir='logs',
    logging_steps=10,
    save_total_limit=1,
    fp16=True
    # weight_decay=0.01,
    # save_total_limit=1,
    # num_train_epochs=5,
    # predict_with_generate=True,
    # fp16=True
)

trainer = Seq2SeqTrainer(
    model, 
    args,
    train_dataset=tokenizedTrainDataset,
    eval_dataset=tokenizedValidationDataset,
    data_collator=collator,
    tokenizer=tokenizer,
    # compute_metrics=computeMetrics,
    optimizers=(optimizer, scheduler)
)

trainer.train()

torch.save(model, "bart.pt")
