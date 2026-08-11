from pymongo import MongoClient


class Database:
    def __init__(self, uri: str, db_name: str):
        self.client = MongoClient(uri, serverSelectionTimeoutMS=5000)
        self.client.admin.command("ping")
        self.db = self.client[db_name]
        self.predictions = self.db["predictions"]
        self.patients = self.db["patients"]
        self.hospitals = self.db["hospitals"]
        self.users = self.db["users"]

    def close(self) -> None:
        self.client.close()
