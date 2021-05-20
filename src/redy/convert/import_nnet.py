#
# Convert from an NNET file into a ViewNetwork
# See `import_nnet`
#

from redy.framework import nodes, views

NNET_COUNTER = 0

class LineReader(object):
    def __init__(self, data):
        self.data = data.splitlines()

    def __call__(self):
        while True:
            assert len(self.data) > 0
            line, self.data = self.data[0], self.data[1:]
            line = line.strip()
            if not line.startswith("//"): break
        return line.split(",")[:-1]

# Based on Marabou's MarabouNetworkNNet loader
# Converts an NNET file (contents) into a ViewNetwork
def import_nnet(data):
    global NNET_COUNTER

    netPrefix = "nnet{}_".format(NNET_COUNTER)
    NNET_COUNTER += 1

    read = LineReader(data)

    # numLayers does not include input layer
    numLayers, inputSize, _, _ = [int(x) for x in read()]

    layerSizes = [int(x) for x in read()]
    assert len(layerSizes) == numLayers+1

    read() # Symmetric? Unused

    inputMinimums = [float(x) for x in read()]
    inputMaximums = [float(x) for x in read()]
    means = [float(x) for x in read()]
    ranges = [float(x) for x in read()]

    weights = []
    biases = []
    for layernum in range(numLayers):
        previousLayerSize = layerSizes[layernum]
        currentLayerSize = layerSizes[layernum + 1]
        # weights
        weights.append([])
        biases.append([])
        # weights
        for i in range(currentLayerSize):
            aux = [float(x) for x in read()]
            weights[layernum].append([])
            for j in range(previousLayerSize):
                weights[layernum][i].append(aux[j])
        # biases
        for i in range(currentLayerSize):
            x = float(read()[0])
            biases[layernum].append(x)

    for i in range(inputSize):
        inputMinimums[i] = (inputMinimums[i] - means[:-1][i]) / ranges[:-1][i]
        inputMaximums[i] = (inputMaximums[i] - means[:-1][i]) / ranges[:-1][i]
    
    layers = [[] for _ in range(len(layerSizes))]
    for i, (imin, imax) in enumerate(zip(inputMinimums, inputMaximums)):
        n = nodes.Node()
        n.name = "{}{:02}_{:02}".format(netPrefix, 0, i)
        n.updateLimit(imin, imax)
        layers[0].append([n])

    for l, (ws, bs) in enumerate(zip(weights, biases), 1):
        lastLayerOutputs = [ns[-1] for ns in layers[l-1]]
        lastLayer = ( l == (len(layers)-1) )

        for n, (wsn, b) in enumerate(zip(ws, bs)):
            name = netPrefix + "{:02}_{:02}".format(l, n)

            n = nodes.NodeSum()
            n.inputs = list(zip(wsn, lastLayerOutputs))
            n.scalar = b

            if not lastLayer:
                n.name = name + "_b"

                nr = nodes.NodeReLU()
                nr.name = name + "_f"
                nr.input = n
                nr.updateLimit(lower=0)

                layers[l].append([n, nr])
            else:
                n.name = name
                layers[l].append([n])

    return views.ViewNetwork(layers)
