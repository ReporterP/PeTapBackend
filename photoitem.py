from PIL import Image
from datetime import datetime

class PhotoItem:
    def __init__(self, photo: Image, reg_datatime: datetime) -> None:
        self.photo = photo
        self.reg_datatime = datetime
        self.isFinished = False
        self.cats_percent = 0
        self.dogs_percent = 0
    
    def finish(self, cats_percent, dogs_percent):
        self.cats_percent = cats_percent
        self.dogs_percent = dogs_percent
        self.isFinished = True