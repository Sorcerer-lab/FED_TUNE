"""A tiny whitespace tokenizer used ONLY by the offline 'tiny-random' check, so the whole
pipeline can be verified without downloading anything. Real runs use Hugging Face tokenizers."""
from collections import Counter

PAD, CLS, SEP, UNK = 0, 1, 2, 3


class SimpleTokenizer:
    pad_token_id = PAD

    def __init__(self, vocab: dict):
        self.vocab = vocab

    @classmethod
    def fit(cls, texts, max_vocab: int = 5000):
        counts = Counter(w for t in texts for w in t.lower().split())
        vocab = {w: i + 4 for i, (w, _) in enumerate(counts.most_common(max_vocab))}
        return cls(vocab)

    @property
    def vocab_size(self) -> int:
        return len(self.vocab) + 4

    def _ids(self, text):
        return [self.vocab.get(w, UNK) for w in text.lower().split()]

    def __call__(self, texts, text_pair=None, truncation=True, max_length=128):
        out = []
        for i, t in enumerate(texts):
            ids = [CLS] + self._ids(t) + [SEP]
            if text_pair is not None:
                ids += self._ids(text_pair[i]) + [SEP]
            out.append(ids[:max_length] if truncation else ids)
        return {"input_ids": out, "attention_mask": [[1] * len(x) for x in out]}
