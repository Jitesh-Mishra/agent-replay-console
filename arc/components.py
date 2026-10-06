"""Data loading, split, and the trained components (router, priority model, retrieval index)."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "tickets_en.csv.gz"
KB_DIR = ROOT / "kb"
SEED = 42
QUEUES = ["Technical Support", "Product Support", "Customer Service", "IT Support", "Billing and Payments",
          "Returns and Exchanges", "Service Outages and Maintenance", "Sales and Pre-Sales", "Human Resources",
          "General Inquiry"]


def load_tickets() -> pd.DataFrame:
    df = pd.read_csv(DATA)
    df["subject"] = df["subject"].fillna("")
    df["text"] = (df["subject"] + "\n" + df["body"]).str.strip()
    return df


def random_split(df: pd.DataFrame, test_size=0.2):
    """LEAKY on this data: paraphrased near-duplicates land on both sides. Kept only as a comparison."""
    tr, te = train_test_split(df, test_size=test_size, random_state=SEED, stratify=df["queue"])
    return tr.reset_index(drop=True), te.reset_index(drop=True)


def near_dup_groups(df: pd.DataFrame, threshold=0.45) -> np.ndarray:
    """Connected components of the TF-IDF cosine >= threshold graph: each component is one family of
    paraphrased tickets. Cached in data/near_dup_groups.csv.gz."""
    cache = ROOT / "data" / "near_dup_groups.csv.gz"
    if cache.exists():
        g = pd.read_csv(cache)
        if len(g) == len(df) and (g["ticket_id"].values == df["ticket_id"].values).all():
            return g["group"].to_numpy()
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    X = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True).fit_transform(df["text"])
    n, rows, cols = len(df), [], []
    for i in range(0, n, 1000):
        S = (X[i:i + 1000] @ X.T).toarray()
        np.fill_diagonal(S[:, i:i + 1000], 0)
        r, c = np.nonzero(S >= threshold)
        rows.append(r + i); cols.append(c)
    r, c = np.concatenate(rows), np.concatenate(cols)
    _, lab = connected_components(coo_matrix((np.ones(len(r)), (r, c)), shape=(n, n)), directed=False)
    pd.DataFrame({"ticket_id": df["ticket_id"], "group": lab}).to_csv(cache, index=False)
    return lab


def split(df: pd.DataFrame, n_splits=5):
    """Group-aware split: no paraphrase family appears in both train and test (about 80/20)."""
    from sklearn.model_selection import StratifiedGroupKFold
    groups = near_dup_groups(df)
    sgk = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    tr_idx, te_idx = next(sgk.split(df, df["queue"], groups))
    return df.iloc[tr_idx].reset_index(drop=True), df.iloc[te_idx].reset_index(drop=True)


def load_kb() -> list[dict]:
    docs = []
    for p in sorted(KB_DIR.glob("*.md")):
        txt = p.read_text()
        meta = dict(re.findall(r"^(\w+):\s*(.+?)(?:\s+#.*)?$", txt.split("---")[1], re.M))
        body = txt.split("---", 2)[2].strip()
        docs.append({"id": p.stem, "queue": meta["queue"], "title": meta["title"], "text": body})
    return docs


# ----------------------------------------------------------------------------- baselines
KEYWORD_RULES = [   # hand-written before looking at test data; first match wins
    ("Service Outages and Maintenance", r"outage|downtime|offline|maintenance|service (is )?down|unavailable"),
    ("Billing and Payments", r"invoice|billing|bill\b|payment|charged|charge|refund|subscription fee"),
    ("Returns and Exchanges", r"return|exchange|replacement|damaged|defective"),
    ("Human Resources", r"payroll|salary|leave|benefit|onboarding|employee|\bhr\b"),
    ("Sales and Pre-Sales", r"pricing|quote|demo|purchase|license cost|trial|pre-sales|sales"),
    ("IT Support", r"vpn|password|laptop|printer|network|wifi|wi-fi|email account|access rights"),
    ("Technical Support", r"error|bug|crash|integration|api|server|database|software"),
    ("Product Support", r"product|feature|app\b|application|update|version|configure|setting"),
    ("Customer Service", r"order|account|delivery|shipping|complaint"),
]


def keyword_route(text: str, default="Technical Support") -> str:
    t = text.lower()
    for q, pat in KEYWORD_RULES:
        if re.search(pat, t):
            return q
    return default


# ----------------------------------------------------------------------------- models
def text_classifier(lowercase=True):
    return Pipeline([("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=200_000,
                                               lowercase=lowercase)),
                     ("lr", LogisticRegression(max_iter=2000, class_weight="balanced"))])


def fit_classifier(X, y, groups=None, lowercase=True, C=None):
    """TF-IDF + logistic regression. C is tuned by GROUPED 3-fold CV on train (plain CV leaks paraphrases)."""
    from sklearn.model_selection import GroupKFold
    pipe = text_classifier(lowercase)
    if C is not None:
        pipe.set_params(lr__C=C)
        return pipe.fit(X, y), {"C": C}
    gs = GridSearchCV(pipe, {"lr__C": [0.1, 0.3, 1.0, 3.0, 10.0]}, scoring="f1_macro", cv=GroupKFold(3), n_jobs=2)
    gs.fit(X, y, groups=groups)
    return gs.best_estimator_, {"C": gs.best_params_["lr__C"], "cv_macro_f1": float(gs.best_score_)}


class Embedder:
    """Offline embedding: TF-IDF -> 256-d LSA, L2-normalised. Used because the default Chroma / HF
    sentence-transformer weights can't be downloaded in this environment."""

    def __init__(self, dims=256):
        self.tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=100_000)
        self.svd = TruncatedSVD(dims, random_state=SEED)

    def fit(self, texts):
        self.svd.fit(self.tfidf.fit_transform(texts))
        return self

    def __call__(self, texts):
        v = self.svd.transform(self.tfidf.transform(list(texts)))
        return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)


@dataclass
class Index:
    """Chroma collections for past tickets (with their answers) and KB docs."""
    embedder: Embedder
    tickets: object
    kb: object


def build_index(train: pd.DataFrame, kb_docs: list[dict], field="text", fraction=1.0, name="v1") -> Index:
    import chromadb
    from chromadb.config import Settings
    emb = Embedder().fit(train[field].tolist() + [d["text"] for d in kb_docs])
    client = chromadb.EphemeralClient(Settings(anonymized_telemetry=False, allow_reset=True))
    t = train.sample(frac=fraction, random_state=SEED) if fraction < 1 else train
    col = client.get_or_create_collection(f"tickets_{name}", metadata={"hnsw:space": "cosine"}, embedding_function=None)
    vecs = emb(t[field])
    for i in range(0, len(t), 4000):
        sl = slice(i, i + 4000)
        col.add(ids=t["ticket_id"].iloc[sl].tolist(), embeddings=vecs[sl].tolist(),
                metadatas=[{"queue": q, "answer": a} for q, a in zip(t["queue"].iloc[sl], t["answer"].iloc[sl])])
    kbc = client.get_or_create_collection(f"kb_{name}", metadata={"hnsw:space": "cosine"}, embedding_function=None)
    kbc.add(ids=[d["id"] for d in kb_docs], embeddings=emb([d["text"] for d in kb_docs]).tolist(),
            metadatas=[{"title": d["title"], "queue": d["queue"]} for d in kb_docs])
    return Index(emb, col, kbc)


def search(index: Index, text: str, k=3, field_text=None):
    q = index.embedder([field_text if field_text is not None else text]).tolist()
    r = index.tickets.query(query_embeddings=q, n_results=k, include=["metadatas", "distances"])
    hits = [{"ticket_id": i, "answer": m["answer"], "queue": m["queue"], "sim": 1 - d}
            for i, m, d in zip(r["ids"][0], r["metadatas"][0], r["distances"][0])]
    rk = index.kb.query(query_embeddings=q, n_results=1, include=["metadatas", "distances"])
    kb = [{"id": i, "title": m["title"], "sim": 1 - d} for i, m, d in zip(rk["ids"][0], rk["metadatas"][0], rk["distances"][0])]
    return hits, kb


class AnswerSim:
    """Similarity between a drafted reply and the historical agent answer (TF-IDF cosine, fit on train answers)."""

    def __init__(self, answers):
        self.v = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True).fit(answers)

    def __call__(self, a, b):
        A, B = self.v.transform(list(a)), self.v.transform(list(b))
        return np.asarray(A.multiply(B).sum(1)).ravel()
