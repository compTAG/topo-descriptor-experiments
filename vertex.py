class Vertex:  # custom vertex object to store x and y coord

    def __init__(self, id, x, y):
        self.id = id
        self.x = x
        self.y = y

    def get_x(self):
        return self.x

    def set_x(self, new_x):
        self.x = new_x

    def get_y(self):
        return self.y

    def set_y(self, new_y):
        self.y = new_y

    def get_id(self):
        return self.id
