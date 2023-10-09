import os 
import pymongo
from dotenv import load_dotenv
load_dotenv()
uri = os.getenv("MONGO_URI")

class MongoDB:
    def __init__(self):
        self.client = pymongo.MongoClient(uri)

    def get_database(self, db_name):
        return self.client[db_name]
    
    def get_collection(self, db_name, collection_name):
        db = self.get_database(db_name)
        return db[collection_name]

    def new_collection(self, new_collection):
        return self.create_Collection(new_collection)

    def find_all(self):
        return list(self.collection.find({}))

    def find_one(self, query):
        return self.collection.find_one(query)

    def insert(self, data):
        if data in self.collection:
            pass
        else:
            uploaded_db = self.get_collection.insert_one(data)
        return print(f"Success: {uploaded_db}")

    def update(self, query, new_data):
        return self.collection.update_one(query, {"$set": new_data})

    def delete(self, query):
        return self.collection.delete_one(query)




def persistent_storage(db_name, collection_name):
    mongo_instance = MongoDB()
    try:
        debate_db = mongo_instance.get_collection(db_name, collection_name)
    except Exception as e:
        debate_logger.error(f"failed to initialise: {e}")
    
    return debate_db





