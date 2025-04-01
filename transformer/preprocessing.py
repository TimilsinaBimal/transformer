# preprocess text into tokens
from .tokenizer.bpe import BytepairEncoding


class Preprocess:
    def train(self, file_path: str, vocab_size: int = 1000):
        bpe = BytepairEncoding()
        bpe.train(file_path, vocab_size)

    def encode(self, text: str) -> list:
        bpe = BytepairEncoding()
        return bpe.encode(text)

    def decode(self, tokens: list):
        bpe = BytepairEncoding()
        return bpe.decode(tokens)
