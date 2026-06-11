"""
backend/database.py
===================
Database layer — MongoDB Atlas + in-memory store
"""

import os
from datetime import datetime

# ─────────────────────────────────────────────────────────────
# IN-MEMORY STORE
# ─────────────────────────────────────────────────────────────
_store: dict = {}


def store_save(image_id: str, record: dict):
    _store[image_id] = record


def store_get(image_id: str):
    return _store.get(image_id)


def store_list():
    return list(_store.values())


def store_exists(image_id: str) -> bool:
    return image_id in _store


# ─────────────────────────────────────────────────────────────
# MONGODB ATLAS
# ─────────────────────────────────────────────────────────────
_mongo_client     = None
_mongo_collection = None
_mongo_connected  = False


def connect_mongodb(uri: str = None) -> bool:
    global _mongo_client, _mongo_collection, _mongo_connected

    if _mongo_connected:
        return True

    mongo_uri = uri or os.getenv("MONGODB_URI", "")
    if not mongo_uri:
        print("  [DB] MONGODB_URI not set — using in-memory store only")
        return False

    try:
        from pymongo import MongoClient
        from pymongo.server_api import ServerApi

        print(f"  [DB] Connecting to MongoDB Atlas...")
        _mongo_client = MongoClient(
            mongo_uri,
            server_api=ServerApi("1"),
            serverSelectionTimeoutMS=5000,
        )
        _mongo_client.admin.command("ping")
        _mongo_collection = _mongo_client["cxr_pacs"]["reports"]
        _mongo_connected  = True
        print(f"  [DB] MongoDB Atlas connected successfully")
        return True

    except Exception as e:
        print(f"  [DB] MongoDB connection failed: {e}")
        _mongo_connected = False
        return False


def mongo_save(record: dict) -> bool:
    if not _mongo_connected or _mongo_collection is None:
        return False
    try:
        record_copy = {k: v for k, v in record.items() if k != "_id"}
        _mongo_collection.update_one(
            {"image_id": record["image_id"]},
            {"$set": record_copy},
            upsert=True,
        )
        return True
    except Exception as e:
        print(f"  [DB] MongoDB save error: {e}")
        return False


def mongo_get(image_id: str):
    if not _mongo_connected or _mongo_collection is None:
        return None
    try:
        record = _mongo_collection.find_one({"image_id": image_id})
        if record:
            record.pop("_id", None)
        return record
    except Exception as e:
        print(f"  [DB] MongoDB get error: {e}")
        return None


def mongo_list():
    if not _mongo_connected or _mongo_collection is None:
        return []
    try:
        return list(_mongo_collection.find({}, {"_id": 0}))
    except Exception as e:
        print(f"  [DB] MongoDB list error: {e}")
        return []


# ─────────────────────────────────────────────────────────────
# UNIFIED FUNCTIONS
# ─────────────────────────────────────────────────────────────
def save_record(image_id: str, record: dict):
    record["image_id"]   = image_id
    record["updated_at"] = datetime.now().isoformat()
    store_save(image_id, record)
    if _mongo_connected:
        mongo_save(record)


def get_record(image_id: str):
    record = store_get(image_id)
    if record:
        return record
    return mongo_get(image_id)


def list_records():
    memory = store_list()
    if memory:
        return memory
    return mongo_list()


def is_mongo_connected() -> bool:
    return _mongo_connected