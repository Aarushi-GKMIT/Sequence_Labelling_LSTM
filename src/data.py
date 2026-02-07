import json
import torch
from torch.utils.data import Dataset, DataLoader

PAD_TOKEN = "<PAD>"
PAD_LABEL = "<PAD>"


def load_data(path):
    data = []
    with open(path) as f:
        for line in f:
            data.append(json.loads(line))
    return data


def build_vocab(data):
    token2id = {PAD_TOKEN: 0}
    label2id = {PAD_LABEL: 0}

    for row in data:
        for tok in row["tokens"]:
            if tok not in token2id:
                token2id[tok] = len(token2id)
        for lab in row["labels"]:
            if lab not in label2id:
                label2id[lab] = len(label2id)

    return token2id, label2id


class NERDataset(Dataset):
    def __init__(self, data, token2id, label2id):
        self.data = data
        self.token2id = token2id
        self.label2id = label2id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        tokens = [self.token2id[t] for t in row["tokens"]]
        labels = [self.label2id[l] for l in row["labels"]]
        return torch.tensor(tokens), torch.tensor(labels)


def collate_fn(batch):
    tokens, labels = zip(*batch)
    max_len = max(len(x) for x in tokens)

    padded_tokens = []
    padded_labels = []

    for t, l in batch:
        pad_len = max_len - len(t)
        padded_tokens.append(
            torch.cat([t, torch.zeros(pad_len, dtype=torch.long)])
        )
        padded_labels.append(
            torch.cat([l, torch.zeros(pad_len, dtype=torch.long)])
        )

    return torch.stack(padded_tokens), torch.stack(padded_labels)


def build_dataloaders(data, token2id, label2id, batch_size=16):
    split = int(0.8 * len(data))
    train_data = data[:split]
    val_data = data[split:]

    train_ds = NERDataset(train_data, token2id, label2id)
    val_ds = NERDataset(val_data, token2id, label2id)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )

    return train_loader, val_loader
