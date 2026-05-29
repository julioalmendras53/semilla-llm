import pickle
from pathlib import Path

import torch

from src.model import MiniLLM
from src.tokenizer import CharTokenizer


DATA_PATH = Path("data/corpus.txt")
CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_PATH = CHECKPOINT_DIR / "semilla_llm.pt"

# Hiperparámetros pequeños para que puedas probarlo en CPU.
BATCH_SIZE = 16
BLOCK_SIZE = 128
MAX_STEPS = 1200
EVAL_INTERVAL = 200
LEARNING_RATE = 3e-4

N_EMBD = 128
N_HEAD = 4
N_LAYER = 4
DROPOUT = 0.1


def get_batch(data, batch_size, block_size, device):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model, train_data, val_data, device):
    model.eval()
    out = {}

    for split, data in [("train", train_data), ("val", val_data)]:
        losses = []
        for _ in range(20):
            xb, yb = get_batch(data, BATCH_SIZE, BLOCK_SIZE, device)
            _, loss = model(xb, yb)
            losses.append(loss.item())
        out[split] = sum(losses) / len(losses)

    model.train()
    return out


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError("No existe data/corpus.txt")

    text = DATA_PATH.read_text(encoding="utf-8")

    if len(text) < BLOCK_SIZE + 10:
        raise ValueError("El corpus es demasiado pequeño. Agrega más texto a data/corpus.txt")

    tokenizer = CharTokenizer.from_text(text)
    encoded = torch.tensor(tokenizer.encode(text), dtype=torch.long)

    n = int(0.9 * len(encoded))
    train_data = encoded[:n]
    val_data = encoded[n:]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Dispositivo: {device}")
    print(f"Caracteres del corpus: {len(text)}")
    print(f"Tamaño del vocabulario: {tokenizer.vocab_size}")

    model = MiniLLM(
        vocab_size=tokenizer.vocab_size,
        block_size=BLOCK_SIZE,
        n_embd=N_EMBD,
        n_head=N_HEAD,
        n_layer=N_LAYER,
        dropout=DROPOUT,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    for step in range(MAX_STEPS + 1):
        if step % EVAL_INTERVAL == 0:
            losses = estimate_loss(model, train_data, val_data, device)
            print(
                f"step {step}: "
                f"train loss {losses['train']:.4f}, "
                f"val loss {losses['val']:.4f}"
            )

        xb, yb = get_batch(train_data, BATCH_SIZE, BLOCK_SIZE, device)

        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    CHECKPOINT_DIR.mkdir(exist_ok=True)

    checkpoint = {
        "model_state": model.state_dict(),
        "tokenizer": tokenizer,
        "config": {
            "vocab_size": tokenizer.vocab_size,
            "block_size": BLOCK_SIZE,
            "n_embd": N_EMBD,
            "n_head": N_HEAD,
            "n_layer": N_LAYER,
            "dropout": DROPOUT,
        },
    }

    with CHECKPOINT_PATH.open("wb") as f:
        pickle.dump(checkpoint, f)

    print(f"Modelo guardado en {CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()
