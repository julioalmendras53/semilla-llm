from dataclasses import dataclass


@dataclass
class CharTokenizer:
    """
    Tokenizador simple por caracteres.
    No usa vocabularios externos. Aprende los caracteres presentes en el corpus.
    """

    stoi: dict
    itos: dict

    @classmethod
    def from_text(cls, text: str):
        chars = sorted(list(set(text)))
        stoi = {ch: i for i, ch in enumerate(chars)}
        itos = {i: ch for ch, i in stoi.items()}
        return cls(stoi=stoi, itos=itos)

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    def encode(self, text: str):
        return [self.stoi[ch] for ch in text if ch in self.stoi]

    def decode(self, ids):
        return "".join(self.itos[int(i)] for i in ids)
