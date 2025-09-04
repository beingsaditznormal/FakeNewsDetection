# PreProcessing.py (robusta & snella)
import re
from typing import List, Dict, Any, Optional
import nltk
from nltk.corpus import stopwords, wordnet
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
import pandas as pd

# ---- bootstrap: SOLO verifica presenza, NIENTE download bloccanti ----
def ensure_nltk():
    def has(pkg): 
        try:
            nltk.data.find(pkg); return True
        except LookupError:
            return False

    missing = []
    if not has("corpora/stopwords"): missing.append("stopwords")
    if not has("corpora/wordnet"):   missing.append("wordnet")

    # POS tagger: prova nuovo nome, altrimenti vecchio
    tagger_ok = has("taggers/averaged_perceptron_tagger_eng") or has("taggers/averaged_perceptron_tagger")
    if not tagger_ok:
        missing.append("averaged_perceptron_tagger_eng")  # nuovo
        # facoltativo: aggiungi anche il vecchio se vuoi assoluta compatibilità
        # missing.append("averaged_perceptron_tagger")

    return missing  # lista dei pacchetti da scaricare a parte

# ---- stopwords (teniamo le negazioni) ----
_NEGATIONS = {"no", "not", "nor", "won't", "wouldn't", "don't", "didn't", "can't",
              "couldn't", "isn't", "aren't", "wasn't", "weren't", "hasn't",
              "haven't", "hadn't", "shouldn't", "mustn't", "mightn't"}

def build_stopwords() -> set:
    try:
        sw = set(stopwords.words("english"))
    except LookupError:
        # se mancassero le risorse, fallback vuoto (evita crash)
        sw = set()
    return sw - _NEGATIONS

# ---- regex precompilate ----
RE_URL      = re.compile(r"https?://\S+|www\.\S+")
RE_TAG      = re.compile(r"[@#]\w+")
RE_HTML     = re.compile(r"&\w+;")
RE_NONASCII = re.compile(r"[^\x00-\x7F]+")
RE_NUM      = re.compile(r"\d+")
RE_SPACES   = re.compile(r"\s+")
RE_TOKEN    = re.compile(r"\b[a-z]{3,}\b")  # parole >= 3 lettere

# ---- normalizzazione leggera ----
def normalize(text: Any) -> str:
    if not isinstance(text, str):
        return ""
    t = text.lower()
    t = RE_URL.sub(" ", t)
    t = RE_TAG.sub(" ", t)
    t = RE_HTML.sub(" ", t)
    t = RE_NONASCII.sub(" ", t)
    t = RE_NUM.sub(" ", t)
    t = t.replace("’", "'").replace("`", "'").replace("'", " ")
    t = RE_SPACES.sub(" ", t).strip()
    return t

def tokenize(text: str) -> List[str]:
    return RE_TOKEN.findall(text)

# ---- lemmatizzazione con POS (tollerante) ----
_lemm = WordNetLemmatizer()

def _to_wn(tag: str):
    return wordnet.ADJ if tag.startswith("J") else \
           wordnet.VERB if tag.startswith("V") else \
           wordnet.ADV if tag.startswith("R") else wordnet.NOUN

def lemmatize(tokens: List[str]) -> List[str]:
    if not tokens:
        return []
    try:
        tagged = pos_tag(tokens)
    except LookupError:
        # tagger mancante: ritorna i token così come sono
        return tokens
    return [_lemm.lemmatize(tok, _to_wn(pos)) for tok, pos in tagged]

# ---- pipeline compatta ----
def preprocess_text(text: Any, stop: Optional[set] = None, do_lemma: bool = True) -> Dict[str, Any]:
    stop = stop or build_stopwords()
    clean = normalize(text)
    toks = [t for t in tokenize(clean) if t not in stop]
    out = {"clean": clean, "tokens": toks}
    if do_lemma:
        out["lemmas"] = [w for w in lemmatize(toks) if w not in stop]
    return out

def preprocess_dataframe(df, text_col: str = "text", do_lemma: bool = True):
    sw = build_stopwords()
    n = len(df)

    print(f"[INFO] Inizio preprocessing su {n} righe (do_lemma={do_lemma})", flush=True)

    results = []
    for i, x in enumerate(df[text_col], start=1):
        results.append(preprocess_text(x, stop=sw, do_lemma=do_lemma))
        if i % 1000 == 0:  # stampa ogni 1000 righe
            print(f"[INFO] Processate {i}/{n} righe...", flush=True)

    res = pd.Series(results)
    df["clean"] = res.apply(lambda d: d["clean"])
    df["tokens"] = res.apply(lambda d: d["tokens"])
    if do_lemma:
        df["lemmas"] = res.apply(lambda d: d["lemmas"])

    print("[INFO] Preprocessing completato!", flush=True)
    return df
