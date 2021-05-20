#
# Equations for Redy representation
#

from enum import Enum, auto

class Equation(object):
    class Comparator(Enum):
        LE = auto()
        EQ = auto()
        GE = auto()

    def __init__(self, terms, comparator, scalar):
        # term0*termC + ... >=/<=/== scalar
        self.terms = terms
        self.comparator = comparator
        self.scalar = scalar

    def duplicate(self):
        e = Equation(self.terms, self.comparator, self.scalar)
        return e

    def translate(self, trans):
        self.terms = [(c, trans(n)) for c, n in self.terms]

