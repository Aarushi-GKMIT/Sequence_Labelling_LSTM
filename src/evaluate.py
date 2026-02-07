import torch
from seqeval.metrics import precision_score, recall_score, f1_score

def evaluate(model, dataloader, label2id):
    model.eval()

    id2label = {v: k for k, v in label2id.items()}

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for tokens, labels in dataloader:
            logits = model(tokens)
            preds = logits.argmax(dim=-1)

            for pred_seq, label_seq in zip(preds, labels):
                pred_tags = []
                true_tags = []

                for p, l in zip(pred_seq, label_seq):
                    if l.item() == 0:  
                        continue

                    pred_tags.append(id2label[p.item()])
                    true_tags.append(id2label[l.item()])

                all_preds.append(pred_tags)
                all_labels.append(true_tags)

    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)

    return precision, recall, f1

