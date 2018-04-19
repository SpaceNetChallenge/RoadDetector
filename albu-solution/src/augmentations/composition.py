import random


class Compose:
    """
    compose transforms from list to apply them sequentially
    """
    def __init__(self, transforms):
        self.transforms = [t for t in transforms if t is not None]

    def __call__(self, **data):
        for t in self.transforms:
            data = t(**data)
        return data


class OneOf:
    """
    with probability prob choose one transform from list and apply it
    """
    def __init__(self, transforms, prob=.5):
        self.transforms = transforms
        self.prob = prob

    def __call__(self, **data):
        if random.random() < self.prob:
            t = random.choice(self.transforms)
            t.prob = 1.
            data = t(**data)
        return data
