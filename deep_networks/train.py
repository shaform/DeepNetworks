class IncrementalAverage(object):
    def __init__(self):
        self.average = 0.0
        self.count = 0

    def add(self, value):
        self.count += 1
        self.average = (value - self.average) / self.count + self.average
