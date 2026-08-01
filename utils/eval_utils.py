"""Answer normalization and relaxed matching for eval (aligned with dataloader preprocess)."""
import re
from utils.dataloader import preprocess_answer


def normalize_answer(s: str) -> str:
    t = preprocess_answer(str(s))
    t = t.lower().strip()
    t = re.sub(r"\s+", " ", t)
    return t


def _negation_present(t: str) -> bool:
    return bool(re.search(r"\b(no|not|without|negative|denies|否认)\b|n't\b", str(t).lower()))


def semantic_match(pred: str, ref: str) -> bool:
    """
    Relaxed match for free-form answers (e.g. Open): normalized equality, then
    containment / token Jaccard; require negation polarity to agree when both are substantive.
    """
    p = normalize_answer(pred)
    r = normalize_answer(ref)
    if p == r:
        return True
    if not p or not r:
        return False
    if len(p) <= 16 and len(r) <= 16:
        if p in r or r in p:
            return min(len(p), len(r)) >= 2
    if _negation_present(p) != _negation_present(r):
        return False
    if p in r or r in p:
        return min(len(p), len(r)) >= 3
    pt, rt = set(p.split()), set(r.split())
    if not pt or not rt:
        return False
    inter = len(pt & rt)
    union = len(pt | rt)
    if union == 0:
        return False
    j = inter / union
    return j >= 0.5 or (inter >= 2 and j >= 0.35)
