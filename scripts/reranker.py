"""BGE-reranker-v2-m3, called directly via transformers (bypasses a FlagEmbedding
bug where its wrapper calls tokenizer.prepare_for_model, removed in newer transformers)."""
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

_model = None
_tokenizer = None

def load():
    global _model, _tokenizer
    if _model is None:
        _tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-reranker-v2-m3")
        _model = AutoModelForSequenceClassification.from_pretrained(
            "BAAI/bge-reranker-v2-m3", torch_dtype=torch.float16
        ).to("cuda")
        _model.eval()
    return _model, _tokenizer

def unload():
    global _model, _tokenizer
    if _model is not None:
        del _model, _tokenizer
        _model = None
        _tokenizer = None
        torch.cuda.empty_cache()

def rerank(query: str, chunks: list[dict], top_n: int = 8) -> list[dict]:
    """chunks: list of dicts with 'content' key. Returns top_n sorted by score desc."""
    model, tokenizer = load()
    pairs = [[query, c["content"]] for c in chunks]
    with torch.no_grad():
        inputs = tokenizer(pairs, padding=True, truncation=True, max_length=512,
                            return_tensors="pt").to("cuda")
        scores = model(**inputs).logits.view(-1).float()
        scores = torch.sigmoid(scores).cpu().tolist()  # normalize to 0-1
    for c, s in zip(chunks, scores):
        c["rerank_score"] = s
    return sorted(chunks, key=lambda c: c["rerank_score"], reverse=True)[:top_n]
