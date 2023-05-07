class Dendrite:

    def __init__(self, id_, length, vector1, vector2, angle):
        self.id = id_
        self.length = round(length * 0.167, 2)
        self.vector1 = vector1
        self.vector2 = vector2
        self.angle = round(angle, 2)

    def __eq__(self, other):
        return self.angle == other.angle

    def __lt__(self, other):
        return self.angle < other.angle

    def __str__(self):
        return "id: {0} , length: {1}, vector1: {2}, vector2: {3}, angle: {4} ".format(self.id, self.length,
                                                                                       self.vector1, self.vector2,
                                                                                      self.angle)
