import torch
import json
from src.model import LSTMTagger
from src.data import build_vocab, load_data


def load_model(model_path, token2id, label2id):
    model = LSTMTagger(
        vocab_size=len(token2id),
        embedding_dim=128,
        hidden_dim=256,
        num_labels=len(label2id),
    )
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model


def predict(tokens, model, token2id, id2label):

    ids = [
        token2id.get(tok, 0)  
        for tok in tokens
    ]

    x = torch.tensor(ids).unsqueeze(0)  

    with torch.no_grad():
        logits = model(x)
        preds = logits.argmax(-1).squeeze(0).tolist()

    labels = [id2label[p] for p in preds]

    return list(zip(tokens, labels))


def main():
    data = load_data("data/dataset.jsonl")
    token2id, label2id = build_vocab(data)
    id2label = {v: k for k, v in label2id.items()}

    model = load_model("artifacts/model.pt", token2id, label2id)

    tokens = input("Enter token sequence ").split()
    predictions = predict(tokens, model, token2id, id2label)

    for tok, lab in predictions:
        print(f"{tok:15} -> {lab}")
    
if __name__ == "__main__":
    main()
