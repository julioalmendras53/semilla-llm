import argparse
import pickle
from pathlib import Path

import torch

from src.model import MiniLLM


CHECKPOINT_PATH = Path("checkpoints/semilla_llm.pt")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="El infinito es")
    parser.add_argument("--max-new-tokens", type=int, default=250)
    parser.add_argument("--temperature", type=float, default=0.9)
    args = parser.parse_args()

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            "No existe checkpoints/semilla_llm.pt. Primero ejecuta: python train.py"
        )

    with CHECKPOINT_PATH.open("rb") as f:
        checkpoint = pickle.load(f)

    tokenizer = checkpoint["tokenizer"]
    config = checkpoint["config"]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = MiniLLM(**config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    start_ids = tokenizer.encode(args.prompt)

    if not start_ids:
        raise ValueError("El prompt no contiene caracteres conocidos por el tokenizador.")

    idx = torch.tensor([start_ids], dtype=torch.long, device=device)

    output = model.generate(
        idx,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )

    print(tokenizer.decode(output[0].tolist()))


if __name__ == "__main__":
    main()
