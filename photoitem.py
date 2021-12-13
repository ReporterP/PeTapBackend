from PIL import Image
from datetime import datetime

class PhotoItem:
    def __init__(self, saved_path: str) -> None:
        self.saved_path = saved_path
        self.reg_datetime = datetime.now()
        self.isFinished = False
        self.cats_percent = 0
        self.dogs_percent = 0
    
    def finish(self, cat, dog):
        self.cats_percent = cat
        self.dogs_percent = dog
        self.isFinished = True

    def toMap(self):
        return {
            "reg_datetime": self.reg_datetime, 
            "isFinished": self.isFinished, 
            "cats_percent": float(self.cats_percent), 
            "dogs_percent": float(self.dogs_percent)
            }