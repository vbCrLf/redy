#
# Nodes for Redy representation
# Node - Base Class
# NodeSum - Weighted Sum Neuron
# NodeReLU - ReLU Neuron
# NodeAbs - Absolute Value Neuron
#

class Node(object):
    def __init__(self):
        self.limit = (None, None)
        self.name = None

    def __repr__(self):
        return "<%s %s [%s]>" % (self.__class__.__name__, self.name, hex(id(self)))

    def updateLimit(self, lower=None, upper=None):
        l,u = self.limit
        if lower is not None and (l is None or lower > l): l = lower
        if upper is not None and (u is None or u > upper): u = upper
        self.limit = l,u

        if lower is not None and upper is not None:
            assert l <= u, (l, u)

    def duplicate(self):
        n = Node()
        n.copyFrom(self)
        return n
    def copyFrom(self, n):
        self.limit = n.limit
        self.name = n.name

    def translate(self, trans):
        pass

    def connectedTo(self):
        return []

class NodeSum(Node):
    def __init__(self):
        super().__init__()
        # List of (coeff, neuron)
        self.inputs = []
        self.scalar = 0

    def duplicate(self):
        n = NodeSum()
        n.copyFrom(self)
        return n
    def copyFrom(self, n):
        super().copyFrom(n)
        self.inputs = n.inputs[:]
        self.scalar = n.scalar

    def translate(self, trans):
        self.inputs = [(c, trans(n)) for c,n in self.inputs]

    def connectedTo(self):
        return [v for c,v in self.inputs]

class NodeReLU(Node):
    def __init__(self):
        super().__init__()
        self.input = None
        self.relaxed = False

    def duplicate(self):
        n = NodeReLU()
        n.copyFrom(self)
        return n
    def copyFrom(self, n):
        super().copyFrom(n)
        self.input = n.input
        self.relaxed = n.relaxed
    
    def translate(self, trans):
        self.input = trans(self.input)

    def connectedTo(self):
        return [self.input]

class NodeAbs(Node):
    def __init__(self):
        super().__init__()
        self.input = None

    def duplicate(self):
        n = NodeAbs()
        n.copyFrom(self)
        return n
    def copyFrom(self, n):
        super().copyFrom(n)
        self.input = n.input
    
    def translate(self, trans):
        self.input = trans(self.input)

    def connectedTo(self):
        return [self.input]

