# dataImport.py
import numpy as np
import pandas as pd
from pymongo import MongoClient
from typing import List, Dict, Any

# ===== Connessione =====
def get_collection(uri: str = "mongodb://localhost:27017/", db: str = "News", coll: str = "rawData"):
    return MongoClient(uri)[db][coll]

# ===== Import CSV + pulizia campo =====
def import_csv(path: str,
               uri: str = "mongodb://localhost:27017/",
               db: str = "News",
               coll: str = "rawData",
               drop_field: str = "Unnamed: 0",
               create_indexes: bool = True) -> Dict[str, Any]:
    """
    Importa un CSV in MongoDB e rimuove (via $unset) un campo indesiderato.
    """
    collection = get_collection(uri, db, coll)

    # Leggi e converti NaN -> None (meglio per Mongo)
    df = pd.read_csv(path)
    records = df.replace({np.nan: None}).to_dict("records")

    if records:
        collection.insert_many(records)
    inserted = len(records)

    # Elimina eventuale colonna spuria
    if drop_field:
        collection.update_many({}, {"$unset": {drop_field: ""}})

    # Indici utili
    if create_indexes:
        collection.create_index("subject")
        collection.create_index("state")
        collection.create_index([("title", "text"), ("text", "text")])  # full-text

    return {"inserted": inserted, "drop_field": drop_field}

# ===== Query esplorative (snelle) =====
def sample(coll, n: int = 5):
    return list(coll.find({}, {"_id": 0}).limit(n))

def counts(coll):
    return {
        "total": coll.count_documents({}),
        "subject_News": coll.count_documents({"subject": "News"}),
        "state_1": coll.count_documents({"state": 1})
    }

def dist_by_subject(coll):
    return list(coll.aggregate([
        {"$group": {"_id": "$subject", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}}
    ]))

def dist_by_year(coll):
    return list(coll.aggregate([
        {"$addFields": {"_date": {"$dateFromString": {"dateString": "$date", "onError": None, "onNull": None}}}},
        {"$project": {"y": {"$year": "$_date"}}},
        {"$group": {"_id": "$y", "count": {"$sum": 1}}},
        {"$sort": {"_id": 1}}
    ]))

def text_len_stats(coll):
    res = list(coll.aggregate([
        {"$project": {"L": {"$strLenCP": {"$ifNull": ["$text", ""]}}}},
        {"$group": {"_id": None, "n": {"$sum": 1}, "avg": {"$avg": "$L"}, "min": {"$min": "$L"}, "max": {"$max": "$L"}}}
    ]))
    return res[0] if res else {}

def search_keyword(coll, kw: str, limit: int = 5):
    q = {"$or": [
        {"title": {"$regex": kw, "$options": "i"}},
        {"text": {"$regex": kw, "$options": "i"}}
    ]}
    hits = list(coll.find(q, {"title": 1, "date": 1}).limit(limit))
    total = coll.count_documents(q)
    return {"total": total, "hits": hits}

def data_quality(coll):
    missing_text = coll.count_documents({
        "$or": [{"text": {"$exists": False}}, {"text": None}, {"text": ""}]
    })
    non_parsable_date = list(coll.aggregate([
        {"$addFields": {"_date": {"$dateFromString": {"dateString": "$date", "onError": None, "onNull": None}}}},
        {"$match": {"_date": None}},
        {"$count": "c"}
    ]))
    return {
        "missing_or_empty_text": missing_text,
        "non_parsable_date": (non_parsable_date[0]["c"] if non_parsable_date else 0)
    }

def previews(coll, n: int = 10, k: int = 140):
    return list(coll.aggregate([
        {"$project": {"_id": 0, "title": 1, "date": 1,
                      "preview": {"$substrCP": [{"$ifNull": ["$text", ""]}, 0, k]}}},
        {"$limit": n}
    ]))

def explore(coll, kw: str = "Trump") -> Dict[str, Any]:
    """
    Raccoglie in un unico dizionario i risultati principali di esplorazione.
    (Niente print: verrà gestito dal main.)
    """
    return {
        "samples": sample(coll, 5),
        "counts": counts(coll),
        "by_subject": dist_by_subject(coll),
        "by_year": dist_by_year(coll),
        "text_stats": text_len_stats(coll),
        "search": search_keyword(coll, kw, 5),
        "data_quality": data_quality(coll),
        "previews": previews(coll, 10, 140)
    }

def import_dataframe(df, uri="mongodb://localhost:27017/", db="News", coll="rawData"):
    col = get_collection(uri, db, coll)
    recs = df.replace({np.nan: None}).to_dict("records")
    if recs:
        col.insert_many(recs, ordered=False)
    return {"inserted": len(recs), "collection": f"{db}.{coll}"}