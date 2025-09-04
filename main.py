# main.py
from pprint import pprint
import pandas as pd
# Importo le mie funzioni
from ImportData import import_dataframe, get_collection, explore

from PreProcessing import ensure_nltk, preprocess_dataframe  # o preprocess_text se lo rinomini
import nltk

missing = ensure_nltk()
if missing:
    print("Scarico risorse NLTK mancanti:", missing)
    for pkg in missing:
        nltk.download(pkg)

# === 1) Leggo il CSV ===
df = pd.read_csv(r"C:\Users\DAVIDEPEDRETTI\Desktop\NLP_news\news.csv")

# === 2) Pre-processing (pulizia + tokenizzazione + lemmatizzazione) ===
df = preprocess_dataframe(df, text_col="text", do_lemma=True)

# === 3) Inserisco in Mongo ===
res_import = import_dataframe(df, db="News", coll="rawData")
print("Import:", res_import)

# === 4) Eseguo le query di esplorazione ===
coll = get_collection(db="News", coll="rawData")
report = explore(coll, kw="Trump")

# === 5) Stampo un report sintetico ===
for k, v in report.items():
    print(f"\n== {k} ==")
    pprint(v)
