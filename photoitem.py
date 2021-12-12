from PIL import Image
from datetime import datetime

class PhotoItem:
    def __init__(self, photo: Image, saved_path: str) -> None:
        self.photo = photo
        self.saved_path = saved_path
        self.reg_datetime = datetime.now()
        self.isFinished = False
        self.cats_percent = 0
        self.dogs_percent = 0
    
    def finish(self, cats_percent, dogs_percent):
        self.cats_percent = cats_percent
        self.dogs_percent = dogs_percent
        self.isFinished = True

    def toMap(self):
        return {
            "reg_datetime": self.reg_datetime, 
            "isFinished": self.isFinished, 
            "cats_percent": self.cats_percent, 
            "dogs_percent": self.dogs_percent
            }