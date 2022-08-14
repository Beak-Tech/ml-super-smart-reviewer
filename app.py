import torch
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("roberta-large") 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
path='model_v2.pt'

model=torch.load(path)

def predict(question, passage):
    sequence = tokenizer.encode_plus(question, passage, return_tensors="pt")['input_ids'].to(device)
    if(len(sequence[0]) > 512):
        return 
    logits = model(sequence)[0]
    probabilities = torch.softmax(logits, dim=1).detach().cpu().tolist()[0]
    proba_yes = round(probabilities[1], 2)
    proba_no = round(probabilities[0], 2)
    return [proba_yes, proba_no]
