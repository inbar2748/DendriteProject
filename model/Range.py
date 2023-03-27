class Range:

    def __init__(self, min_, max_, id_, angle, length, dendrite):
        self.min = round(min_, 2)
        self.max = round(max_, 2)
        self.id = id_
        self.angle = round(angle, 2)
        self.length = round(length, 2)
        self.dendrite = dendrite

    def __str__(self):
        return "Dendrite: {0} , \nrange: {1}---> {2}".format(self.dendrite, self.min, self.max)