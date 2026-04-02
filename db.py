from pymongo import MongoClient
from dotenv import load_dotenv
import os
import certifi

load_dotenv()

_client = None

def get_db():
    global _client
    if _client is None:
        uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
        _client = MongoClient(
            uri,
            tls=True,
            tlsCAFile=certifi.where(),
            tlsAllowInvalidCertificates=True,
            serverSelectionTimeoutMS=30000,
        )
    return _client[os.getenv("MONGODB_DB", "attendance_db")]

def get_students_col():   return get_db()["students"]
def get_attendance_col(): return get_db()["attendance"]
def get_subjects_col():   return get_db()["subjects"]
def get_classes_col():    return get_db()["classes"]
def get_model_col():      return get_db()["model"]
