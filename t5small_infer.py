import re
import html
import torch
import joblib
import pandas as pd
import evaluate
from tqdm import tqdm
from datasets import load_dataset
from transformers import T5Tokenizer
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder

dataset = load_dataset("social_bias_frames")

regexPatterns = {
    "RT": r"RT @[^:]+:",
    "URL": r"https?://\S+|www\.\S+",
    "PUNC": r"[.,?!#@;]",
    "SPC_PUNC": r"[-']"
}

def preprocessText(text):
    text = html.unescape(text)
    for key, reg in regexPatterns.items():
        text = re.sub(reg, ' ' if key == "SPC_PUNC" else '', text)
    return text.strip()

tokenizerParams = {"max_length": 512, "truncation": True, "padding": "max_length", "return_tensors": "pt"}

class EmphasizedConclusionDataset(Dataset):
    def __init__(self, srcSamples, tgtSamples, tokenizer, maxLength=512):
        assert len(srcSamples) == len(tgtSamples)
        self.srcSamples = srcSamples
        self.tgtSamples = tgtSamples
        self.tokenizer = tokenizer
        self.maxLength = maxLength

    def __len__(self):
        return len(self.srcSamples)

    def __getitem__(self, index):
        srcSample = self.srcSamples[index]
        tgtSample = self.tgtSamples[index]
        srcEncoding = self.tokenizer.encode_plus(srcSample, **tokenizerParams)
        tgtEncoding = self.tokenizer.encode_plus(tgtSample, **tokenizerParams)
        return {
            "input_ids": srcEncoding["input_ids"].flatten(),
            "attention_mask": srcEncoding["attention_mask"].flatten(),
            "labels": tgtEncoding["input_ids"].flatten()
        }
    
def preprocessSrcText(sample):
    excludedCols = ["WorkerId", "HITId", "annotatorPolitics"]
    srcText = f"[POST] {preprocessText(sample['post'])} [POST]"
    for colName in sample.keys()[: 14]:
        if colName not in excludedCols:
            srcText += f' {colName}:{sample[colName]}'
    return srcText

def getDataset(partition, size, tokenizer):
    dataframe = pd.DataFrame(dataset[partition].select(range(size)))
    srcSamples, tgtSamples = [], []
    for _, sample in dataframe.iterrows():
        tgtText = sample["targetStereotype"]
        if len(tgtText) == 0:
            continue
        tgtSamples.append(tgtText)
        srcText = preprocessSrcText(sample)
        srcSamples.append(srcText)
    return EmphasizedConclusionDataset(srcSamples, tgtSamples, tokenizer)

modelName = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(modelName)
model = torch.load("t5small.pt")

testSplitSize = 10000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

def genText(text, model, tokenizer, device, maxLength=512):
    model.eval()
    inputTokens = tokenizer.encode(text, **tokenizerParams).to(device)
    outputs = model.generate(inputTokens, max_length=maxLength)
    return tokenizer.decode(outputs[0], skip_special_tokens=True, **tokenizerParams)

def getSamples(partition, size):
    dataframe = pd.DataFrame(dataset[partition].select(range(size)))
    srcSamples, tgtSamples = [], []
    excludedCols = ["WorkerId", "HITId"]
    for _, sample in dataframe.iterrows():
        tgtText = sample["targetStereotype"]
        if len(tgtText) == 0:
            continue
        tgtSamples.append(tgtText)
        srcText = f"[POST] {preprocessText(sample['post'])} [POST]"
        for colName in sample.keys()[: 14]:
            if colName not in excludedCols:
                srcText += f' {colName}:{sample[colName]}'
        srcSamples.append(srcText.rstrip())
    return srcSamples, tgtSamples

def computeBert(output):
    predictions = output["gen"]
    references = output["tgt"]
    nonEmptyPairs = [(pred, ref) for pred, ref in zip(predictions, references) if len(ref) > 0]
    nonEmptyPreds, nonEmptyRefs = zip(*nonEmptyPairs)
    inputs = {"predictions": nonEmptyPreds, "references": nonEmptyRefs}
    bertF1s = evaluate.load("bertscore").compute(**inputs, lang="en")["f1"]
    return sum(bertF1s)/len(bertF1s) if bertF1s else 0

def computeMetrics(output):
    inputs = {"predictions": output["gen"], "references": output["tgt"]}
    return {"bleu": evaluate.load("bleu").compute(**inputs, max_order=2)["bleu"],
            "rougeL": evaluate.load("rouge").compute(**inputs)["rougeL"],
            "bert": computeBert(output)}

classificationModel = joblib.load("hate_speech_classification.pkl")

def inference(dataframe):
    tmpDataframe = dataframe.drop(columns=["post", "targetMinority", "targetCategory", "targetStereotype"], axis=1)
    categoricalCols = tmpDataframe.select_dtypes(include=["object"]).columns.tolist()
    labelEncoders = {col: LabelEncoder().fit(tmpDataframe[col]) for col in categoricalCols}
    for col, encoder in labelEncoders.items():
        tmpDataframe[col] = encoder.transform(tmpDataframe[col])
    classYN = classificationModel.predict(tmpDataframe)
    model.eval()
    genTexts = []
    for i, sample in tqdm(dataframe.iterrows()):
        if len(classYN[i]) == 0:
            genTexts.append("")
            continue
        srcText = preprocessSrcText(sample)
        genTexts.append(genText(srcText, model, tokenizer, device))
    return {"src": dataframe["post"].tolist(), "tgt": dataframe["targetStereotype"].tolist(), "gen": genTexts}

testDataframe = pd.DataFrame(dataset["test"].select(range(testSplitSize)))
output = inference(testDataframe)
print(output)

metrics = computeMetrics(output)
print(metrics)