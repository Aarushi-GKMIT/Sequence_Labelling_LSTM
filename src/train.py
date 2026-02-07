import torch
import torch.nn as nn
from src.data import load_data, build_vocab, build_dataloaders
from src.model import LSTMTagger
from src.evaluate import evaluate
import os


def main():
    data = load_data("data/dataset.jsonl")
    token2id, label2id = build_vocab(data)
    train_loader, val_loader = build_dataloaders(data, token2id, label2id)

    model = LSTMTagger(
        vocab_size=len(token2id),
        embedding_dim=128,
        hidden_dim=256,
        num_labels=len(label2id),
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    num_epochs = 20

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0

        for tokens, labels in train_loader:
            optimizer.zero_grad()
            logits = model(tokens)

            loss = criterion(
                logits.view(-1, logits.size(-1)),
                labels.view(-1)
            )

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        p, r, f1 = evaluate(model, val_loader, label2id)

        print(
            f"Epoch {epoch} | "
            f"Loss: {total_loss:.4f} | "
            f"P: {p:.3f} R: {r:.3f} F1: {f1:.3f}"
        )

    os.makedirs("artifacts", exist_ok=True)
    torch.save(model.state_dict(), "artifacts/model.pt")


if __name__ == "__main__":
    main()
