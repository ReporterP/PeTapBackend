import string
import random


class Database:
    def __init__(self) -> None:
        self.database = {}
    
    def add(self, data):
        key = self.__generate_unique_key()
        self.database[key] = data
        return key

    def get(self, key):
        return self.database[key]

    def __generate_unique_key(self):
        letters = string.ascii_letters + string.digits
        key = ""
        for i in range(16):
            key += letters[random.randint(0, len(letters))]
        if key in self.database.keys():
            return self.__generate_unique_key() 
        return key