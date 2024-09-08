class Product:
    def __init__(self, label=None, image_path=None, image=None, embedding=None):
        self.label = label
        self.image_path = image_path
        self.image = image
        self.embedding = embedding
        self.image_to_predict = None
