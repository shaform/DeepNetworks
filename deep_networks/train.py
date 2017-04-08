"""Training utilities"""


class IncrementalAverage(object):
    """IncrementalAverage

    compute average incrementally
    """

    def __init__(self, average=0.0, count=0):
        self.average = average
        self.count = count

    def add(self, value):
        """add one value towards the average

        :param value: new value
        """
        self.count += 1
        self.average = (value - self.average) / self.count + self.average
