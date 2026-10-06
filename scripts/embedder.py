"""BGE-M3 dense + sparse embedding wrapper. Loads on demand, unload() frees VRAM."""
from FlagEmbedding import BGEM3FlagModel

_model = None

def load():
    global _model
    if _model is None:
        _model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True, device="cuda")
    return _model

def unload():
    global _model
    if _model is not None:
        del _model
        _model = None
        import torch; torch.cuda.empty_cache()

def embed(texts: list[str]) -> dict:
    """Returns {'dense': [[float]*1024], 'sparse': [{token_id: weight}]}"""
    m = load()
    out = m.encode(texts, return_dense=True, return_sparse=True, return_colbert_vecs=False)
    dense = [v.tolist() for v in out["dense_vecs"]]
    sparse = [{str(k): float(v) for k, v in d.items()} for d in out["lexical_weights"]]
    return {"dense": dense, "sparse": sparse}
