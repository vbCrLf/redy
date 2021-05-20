#
# Convenient containers for neurons
# ViewIO - Generic model with nodes, equations, inputs and outputs
# ViewNetwork - More strict model with layers and without equations
#

class ViewIO(object):
    def __init__(self, nodes, inputs, outputs, equations):
        self.nodes = nodes
        self.inputs = inputs
        self.outputs = outputs
        self.equations = equations
        self.sanity()
    
    def sanity(self):
        assert all((x in self.nodes) for x in self.inputs+self.outputs)
        assert all((v in self.nodes) for e in self.equations for c,v in e.terms)

class ViewNetwork(object):
    def __init__(self, layers):
        self.layers = layers
        self.sanity()

    def sanity(self):
        nodes = set(self.nodes())
        for l in self.layers:
            for n in l:
                for node in n:
                    assert nodes.issuperset(node.connectedTo())

    def layer(self, layerId):
        assert 0 <= layerId < self.layerCount()
        return self.layers[layerId]

    def layerCount(self):
        return len(self.layers)

    def layerSize(self, layer):
        return len(self.layers[layer])

    def nodes(self):
        return [n for layer in self.layers for ns in layer for n in ns]

    def duplicate(self, suffix=""):
        nodesTable = { n: n.duplicate() for n in self.nodes() }
        for n in nodesTable.values():
            if n.name is not None: n.name += suffix
        trans = nodesTable.get

        [n.translate(trans) for n in nodesTable.values()]
        nLayers = [[[trans(n) for n in ns] for ns in l] for l in self.layers]

        return ViewNetwork(nLayers)

    def toViewIO(self):
        return ViewIO(self.nodes(), [ns[0] for ns in self.layers[0]], [ns[-1] for ns in self.layers[-1]], [])
