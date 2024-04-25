import re
import html
import torch
import joblib
import evaluate
from tqdm import tqdm
import pandas as pd
from pprint import pprint
from torch.utils.data import Dataset
from datasets import load_dataset
from sklearn.preprocessing import LabelEncoder
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling

dataset = load_dataset("social_bias_frames")

regex_patterns = {
    "RT": r"RT @[^:]+:",
    "URL": r"https?://\S+|www\.\S+",
    "PUNC": r"[.,?!#@;/-\\\"\']",
    "SPC_PUNC": r"[-']"
}

def preprocessDataframe(dataframe):
    for i, post in enumerate(dataframe["post"].to_list()):
        post = html.unescape(post)
        for key, reg in regex_patterns.items():
            delim = ' ' if key == "SPC_PUNC" else ''
            post = re.sub(reg, delim, post)
        post = post.strip()
        dataframe.loc[i, "post"] = post

class ArgumentDataset(Dataset):
    def __init__(self, srcSamples, tgtSamples, tokenizer, maxLength=256):
        assert len(srcSamples) == len(tgtSamples)
        self.srcSamples = srcSamples
        self.tgtSamples = tgtSamples
        self.tokenizer = tokenizer
        self.maxLength = maxLength
        self.tokenizerParams = {"add_special_tokens": True, "max_length": self.maxLength,
                                "padding": "max_length", "truncation": True, "return_tensors": "pt"}

    def __len__(self):
        return len(self.srcSamples)

    def __getitem__(self, index):
        srcEncoding = self.tokenizer(self.srcSamples[index], **self.tokenizerParams)
        tgtEncoding = self.tokenizer(self.tgtSamples[index], **self.tokenizerParams)
        return {"input_ids": srcEncoding["input_ids"].squeeze(0),
                "attention_mask": srcEncoding["attention_mask"].squeeze(0),
                "labels": tgtEncoding["input_ids"].squeeze(0)}

def getTokenizedDataset(partition, size, tokenizer, test=False):
    dataframe = pd.DataFrame(dataset[partition] if size is None else dataset[partition].select(range(size)))
    dataframe.drop_duplicates(subset=["post", "targetStereotype"], keep="first", inplace=True)   
    dataframe.dropna()  
    preprocessDataframe(dataframe)
    srcSamples, tgtSamples = [], []
    for _, sample in dataframe.iterrows():
        tgtText = sample["targetStereotype"]
        if pd.isna(tgtText) or len(tgtText) == 0:
            continue
        tgtSamples.append(tgtText)
        srcText = sample["post"]
        if not test: srcText += f'[SEP] {sample["targetStereotype"]}'
        srcSamples.append(srcText.rstrip())
    return ArgumentDataset(srcSamples, tgtSamples, tokenizer)

modelName = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(modelName, padding_side="left")
model = GPT2LMHeadModel.from_pretrained(modelName)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)

trainDataset = getTokenizedDataset("train", 5000, tokenizer)
valDataset = getTokenizedDataset("validation", 500, tokenizer, True)

batchSize = 32

trainArgs = TrainingArguments(
    output_dir="results",
    evaluation_strategy="epoch",
    logging_strategy="epoch",
    num_train_epochs=42,
    per_device_train_batch_size=batchSize,
    per_device_eval_batch_size=batchSize,
    weight_decay=0.01,
    logging_dir='logs',
    logging_steps=10,
    save_total_limit=1,
    fp16=True
)

trainer = Trainer(
    model=model,
    args=trainArgs,
    train_dataset=trainDataset,
    eval_dataset=valDataset,
    data_collator=collator,
    optimizers=(optimizer, scheduler)
)

trainer.train()

torch.save(model, "gpt.pt")

classificationModel = joblib.load("hate_speech_classification.pkl")

def inference(dataframe):
    tmpDataframe = dataframe.drop(columns=["post", "targetMinority", "targetCategory", "targetStereotype"], axis=1)
    categoricalCols = tmpDataframe.select_dtypes(include=["object"]).columns.tolist()
    labelEncoders = {col: LabelEncoder().fit(tmpDataframe[col]) for col in categoricalCols}
    for col, encoder in labelEncoders.items():
        tmpDataframe[col] = encoder.transform(tmpDataframe[col])
    out = classificationModel.predict(tmpDataframe)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    genTexts = []
    tokenizerParams = {"add_special_tokens": True, "max_length": 256, "padding": "max_length",
                       "truncation": True, "return_tensors": "pt"}
    for i, sample in tqdm(dataframe.iterrows()):
        if len(out[i]) == 0:
            genTexts.append("")
            continue
        inputs = {k: v.to(device) for k, v in tokenizer(sample["post"], **tokenizerParams).items()}
        outputs = model.generate(**inputs, max_length=512, top_k=250, top_p=0.2, 
                                 pad_token_id=model.config.eos_token_id, length_penalty=-0.2, 
                                 remove_invalid_values=False)
        genTexts.append(tokenizer.decode(outputs[0], skip_special_tokens=True))
    return {"src": dataframe["post"].tolist(), "tgt": dataframe["targetStereotype"].tolist(), "gen": genTexts}

testDataframe = pd.DataFrame(dataset["test"].select(range(10)))
preprocessDataframe(testDataframe)

output = inference(testDataframe)

print(output["gen"])
print(output["tgt"])

nEmpty = 0
nCorrect = 0
nWrong = 0
for i in range(len(output["tgt"])):
    if len(output["tgt"][i]) == 0:
        nEmpty += 1
        if len(output["gen"][i]) == 0:
            nCorrect += 1
        else:
            nWrong += 1
print(nEmpty, nCorrect, nWrong)

def computeMetrics(output):
    sample = {"predictions": output["gen"], "references": output["tgt"]}
    bertF1s = evaluate.load("bertscore").compute(**sample, lang="en")["f1"]
    bertF1Avg = sum(bertF1s)/len(bertF1s)
    print(f"Debug: bertF1Avg({bertF1Avg})")
    bleuScore = evaluate.load("bleu").compute(**sample, max_order=2)["bleu"]
    print(f"Debug: bleu2({bleuScore})")
    rougeL = evaluate.load("rouge").compute(**sample)["rougeL"]
    print(f"Debug: rougeL({rougeL})")
    return {"bertscore": bertF1Avg,
            "bleu": bleuScore,
            "rougeL": rougeL}

metrics = computeMetrics(output)
pprint(metrics)

