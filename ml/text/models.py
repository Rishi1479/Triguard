"""Text classifiers: TF-IDF + LogisticRegression and the base paper's Bi-LSTM (PyTorch)."""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from ml.common import config

_TOKEN_RE = re.compile(r"[a-z0-9₹$]+")


class TfidfLR:
    """Word 1-2 + char 3-5 TF-IDF features with balanced Logistic Regression. Fit on train only."""

    def __init__(self) -> None:
        self.word = TfidfVectorizer(ngram_range=config.TFIDF_WORD_NGRAMS, sublinear_tf=True, min_df=2,
                                    max_features=config.TFIDF_MAX_FEATURES, lowercase=True)
        self.char = TfidfVectorizer(analyzer="char_wb", ngram_range=config.TFIDF_CHAR_NGRAMS, sublinear_tf=True,
                                    min_df=2, max_features=config.TFIDF_MAX_FEATURES, lowercase=True)
        self.clf = LogisticRegression(class_weight="balanced", max_iter=2000, C=4.0, random_state=config.SEED)

    def _x(self, texts: list[str], fit: bool = False):
        if fit:
            return hstack([self.word.fit_transform(texts), self.char.fit_transform(texts)]).tocsr()
        return hstack([self.word.transform(texts), self.char.transform(texts)]).tocsr()

    def fit(self, texts: list[str], y) -> "TfidfLR":
        """Fit vectorizers and classifier on the training split."""
        self.clf.fit(self._x(list(texts), fit=True), np.asarray(y))
        return self

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        """Raw (uncalibrated) P(scam) per text."""
        return self.clf.predict_proba(self._x(list(texts)))[:, 1]


def tokenize(text: str) -> list[str]:
    """Lower-case alphanumeric tokenizer used by the Bi-LSTM."""
    return _TOKEN_RE.findall(text.lower())


class Vocab:
    """Token -> id map built from the training split only (0 = pad, 1 = unk)."""

    def __init__(self, token_to_id: dict[str, int]) -> None:
        self.token_to_id = token_to_id

    @classmethod
    def build(cls, texts: list[str], size: int = config.BILSTM_VOCAB_SIZE) -> "Vocab":
        """Most frequent ``size`` tokens in ``texts``."""
        counts = Counter(t for x in texts for t in tokenize(x))
        mapping = {tok: i + 2 for i, (tok, _) in enumerate(counts.most_common(size - 2))}
        return cls(mapping)

    def encode(self, text: str, max_len: int = config.BILSTM_MAX_LEN) -> list[int]:
        """Token ids truncated and right-padded to ``max_len``."""
        ids = [self.token_to_id.get(t, 1) for t in tokenize(text)][:max_len]
        return ids + [0] * (max_len - len(ids))


def _build_bilstm_module():
    import torch
    from torch import nn

    class BiLSTM(nn.Module):
        """Embedding -> Bi-LSTM -> max-pool -> dense -> logit (Pillai 2023 style, text only)."""

        def __init__(self, vocab_size: int) -> None:
            super().__init__()
            self.emb = nn.Embedding(vocab_size, config.BILSTM_EMBED_DIM, padding_idx=0)
            self.lstm = nn.LSTM(config.BILSTM_EMBED_DIM, config.BILSTM_HIDDEN, batch_first=True, bidirectional=True)
            self.drop = nn.Dropout(0.4)
            self.out = nn.Linear(2 * config.BILSTM_HIDDEN, 1)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            h, _ = self.lstm(self.emb(x))
            return self.out(self.drop(h.max(dim=1).values)).squeeze(-1)

    return BiLSTM


class BiLSTMClassifier:
    """Trainer/predictor wrapper around the Bi-LSTM (CPU friendly)."""

    def __init__(self) -> None:
        self.vocab: Vocab | None = None
        self.net = None

    def _tensor(self, texts: list[str]):
        import torch
        return torch.tensor([self.vocab.encode(t) for t in texts], dtype=torch.long)

    def fit(self, texts: list[str], y, val_texts: list[str], val_y) -> "BiLSTMClassifier":
        """Train with weighted BCE; keep the epoch with the best validation PR-AUC (early stopping)."""
        import torch
        from sklearn.metrics import average_precision_score
        from torch import nn

        torch.manual_seed(config.SEED)
        self.vocab = Vocab.build(list(texts))
        self.net = _build_bilstm_module()(len(self.vocab.token_to_id) + 2)
        x, yt = self._tensor(list(texts)), torch.tensor(np.asarray(y), dtype=torch.float32)
        pos = float(yt.sum())
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((len(yt) - pos) / max(pos, 1.0)))
        opt = torch.optim.Adam(self.net.parameters(), lr=config.BILSTM_LR)
        best, best_state, bad = -1.0, None, 0
        g = torch.Generator().manual_seed(config.SEED)
        for _ in range(config.BILSTM_EPOCHS):
            self.net.train()
            perm = torch.randperm(len(x), generator=g)
            for i in range(0, len(x), config.BILSTM_BATCH):
                idx = perm[i: i + config.BILSTM_BATCH]
                opt.zero_grad()
                loss_fn(self.net(x[idx]), yt[idx]).backward()
                opt.step()
            score = average_precision_score(val_y, self.predict_proba(val_texts))
            if score > best:
                best, bad = score, 0
                best_state = {k: v.clone() for k, v in self.net.state_dict().items()}
            else:
                bad += 1
                if bad >= config.BILSTM_PATIENCE:
                    break
        self.net.load_state_dict(best_state)
        return self

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        """Raw P(scam) per text."""
        import torch
        self.net.eval()
        x, out = self._tensor(list(texts)), []
        with torch.no_grad():
            for i in range(0, len(x), 256):
                out.append(torch.sigmoid(self.net(x[i: i + 256])).numpy())
        return np.concatenate(out)

    def save(self, folder: Path) -> None:
        """Write weights and vocabulary to ``folder``."""
        import torch
        folder.mkdir(parents=True, exist_ok=True)
        torch.save(self.net.state_dict(), folder / "model.pt")
        (folder / "vocab.json").write_text(json.dumps(self.vocab.token_to_id))
