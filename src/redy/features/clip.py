#
# A function for clipping the start/end of the network
#

from collections import namedtuple

from redy.framework.nodes import Node

Range = namedtuple("Range", ("firstLayer", "firstMode", "lastLayer", "lastMode"), defaults=(0, 0, None, 0))

# Given a network and a range, clips layers before firstLayer and after lastLayer.
# firstMode/lastMode allows choosing where to clip in the layer (WS or ReLU for example)
def clipNetwork(net, rng): 
    if rng.lastLayer is None:
        rng = Range(rng.firstLayer, rng.firstMode, net.layerCount()-1, rng.lastMode)

    # Clip the network
    layers = net.layers[rng.firstLayer:rng.lastLayer+1]
    bdrs = [n[rng.firstMode] for n in layers[0]]

    # Transform the first node (which might be ReLU or sum) into a plain node
    inputs = {n: Node() for n in bdrs}
    for b,n in inputs.items():
        n.copyFrom(b)
        if b.name is not None: n.name = b.name + "_bdr"
    layers[ 0] = [n[rng.firstMode+1:] for n in layers[0]]
    layers[-1] = [n[:rng.lastMode+1] for n in layers[-1]]

    [inputs[n].copyFrom(n) for n in bdrs]
    [n.insert(0, inputs[b]) for n, b in zip(layers[0], bdrs)]

    def translate(neuron):
        if neuron in inputs: return inputs[neuron]
        return neuron
    [n.translate(translate) for n in net.nodes()]

    # Finalize
    net.layers = layers
    net.sanity()

    return net

