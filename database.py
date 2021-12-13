import string
import random


class Database:
    def __init__(self) -> None:
        self._database = {}
    
    def add(self, data):
        key = self.__generate_unique_key()
        self._database[key] = data
        return key

    def get(self, key):
        return self._database[key]

    def keys(self):
        return self._database.keys()

    def values(self):
        return self._database.values()

    def __generate_unique_key(self):
        letters = string.ascii_letters + string.digits
        key = ""
        for i in range(16):
            key += letters[random.randint(0, len(letters) - 1)]
        if key in self._database.keys():
            return self.__generate_unique_key() 
        return key