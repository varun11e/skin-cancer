from pymongo import ASCENDING, DESCENDING, MongoClient


class Database:
    def __init__(self, uri: str, db_name: str):
        self.client = MongoClient(uri, serverSelectionTimeoutMS=5000, connectTimeoutMS=5000)
        self.client.admin.command("ping")
        self.db = self.client[db_name]
        self.predictions = self.db["predictions"]
        self.patients = self.db["patients"]
        self.hospitals = self.db["hospitals"]
        self.users = self.db["users"]
        self._ensure_indexes()

    def _ensure_indexes(self) -> None:
        self.users.create_index([("username", ASCENDING)], unique=True)
        self.predictions.create_index([("username", ASCENDING), ("timestamp", DESCENDING)])
        self.predictions.create_index([("username", ASCENDING), ("prediction", ASCENDING)])
        self.patients.create_index([("username", ASCENDING), ("phone", ASCENDING)])
        self.hospitals.create_index([("city", ASCENDING)])

    def close(self) -> None:
        self.client.close()
