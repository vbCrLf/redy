#
# Functions for amending a network. See examples
#

from redy.framework.nodes import Node
from redy.framework.equations import Equation
from redy.framework.views import ViewIO

# Given network `net` and a list of neurons `neurons` = {(layer, neuron): f, ...}
# where f \in {active, inactive, nofunc}
# replaces the given neurons' activation function to the function given
def modify(net, neurons):
    for (li, ni), f in neurons.items():
        neuron = net.layers[li][ni]
        if f == "active":
            neuront = [neuron[0]]
        elif f == "inactive":
            n = Node()
            n.updateLimit(0, 0)
            neuront = [neuron[0], n]
        elif f == "nofunc":
            n = Node()
            n.copyFrom(neuron[1])
            neuront = [neuron[0], n]
        else:
            assert False, "Unknown function %s" % (f, )
        if neuron[-1].name is not None: neuront[-1].name = neuron[-1].name + "_" + f
        net.layers[li][ni] = neuront

        def translate(n):
            if n == neuron[-1]: return neuront[-1]
            else: return n
        if li < net.layerCount()-1:
            for n in net.layers[li+1]:
                n[0].translate(translate)

    # Make sure nothing wrong was done
    net.sanity()

    return net

def _join(net, mod, neurons):
    # Remove the identical neurons from the modified network and join it with the original
    firstDupLayer = min(l for l, n in neurons)
    assert firstDupLayer < net.layerCount()
    dup = mod.layers[firstDupLayer:]
    table = {}
    for i in range(len(dup[0])):
        if (firstDupLayer, i) not in neurons:
            table[dup[0][i][-1]] = net.layers[firstDupLayer][i][-1]
        else:
            orig = dup[0][i]
            repl = net.layers[firstDupLayer][i]
            table[orig[0]] = repl[0]
            orig[0] = repl[0]
    dup[0] = [ns for i, ns in enumerate(dup[0]) if (firstDupLayer, i) in neurons]
    [node.translate(lambda x: ( table[x] if x in table else x )) for l in mod.layers for n in l for node in n]

    return dup

# Given an original and modified networks: joins them and compare their output
# `net` - original network
# `mod` - modified network
# `neruons` - a list of modified neurons, used to choose the join point
# `comparator` - how should the output be compared? (gt or lt)
# `output` - output number to be compared
# `epsilon` - the maximum imprecision
def compareExact(net, mod, neurons, comparator, output, epsilon):
    assert comparator in ["gt", "lt"]

    # Join the networks
    dup = _join(net, mod, neurons)

    # Remove excess outputs and compare
    assert output < len(net.layers[-1])
    assert output < len(mod.layers[-1])
    net.layers[-1] = net.layers[-1][output:output+1]
    mod.layers[-1] = mod.layers[-1][output:output+1]
    c = 1 if comparator == "gt" else -1
    equations = [Equation([
        (c, net.layers[-1][0][-1]),
        (-1*c, mod.layers[-1][0][-1]),
    ], Equation.Comparator.GE, epsilon)]
    outputs = [n.layers[-1][0][-1] for n in [net, mod]]

    # Sanity
    nodes = set()
    [nodes.add(node) for l in net.layers for n in l for node in n]
    [nodes.add(node) for l in dup for n in l for node in n]
    for node in nodes: assert nodes.issuperset(node.connectedTo())

    # Return
    inputs = [ns[0] for ns in net.layers[0]]
    return ViewIO(nodes, inputs, outputs, equations)

# Given an original and modified networks: joins them
# `net` - original network
# `mod` - modified network
# `neruons` - a list of modified neurons, used to choose the join point
def join(net, mod, neurons):
    # Join the networks
    dup = _join(net, mod, neurons)

    # Sanity
    nodes = set()
    [nodes.add(node) for l in net.layers for n in l for node in n]
    [nodes.add(node) for l in dup for n in l for node in n]
    for node in nodes: assert nodes.issuperset(node.connectedTo())

    # Return
    inputs = [ns[0] for ns in net.layers[0]]
    outputs = [ns[-1] for ns in (net.layers[-1] + mod.layers[-1])]
    return ViewIO(nodes, inputs, outputs, [])

# Given an original and modified networks: joins them and compare their output
# (result-preserving wise, minimum class wins)
# `net` - original network
# `mod` - modified network
# `neruons` - a list of modified neurons, used to choose the join point
# `output` - output number to be compared
# `counteroutput` - output to compare with
# `epsilon` - the maximum imprecision
# This function generates a query which looks for inputs for which
# `output` is the winning class in the original, but `counteroutput` wins
# `output` in the modified network.
def compareMinimum(net, mod, neurons, output, counteroutput, epsilon):
    # Join the networks
    dup = _join(net, mod, neurons)

    # Apply equations
    assert counteroutput != output
    equations = []

    # 1. Assert `output` is the winner
    winner = net.layers[-1][output][-1]
    for ni, n in enumerate(net.layers[-1]):
        n = n[-1]
        if ni == output: continue
        equations += [Equation([
            (1, winner),
            (-1, n),
        ], Equation.Comparator.LE, -epsilon)]

    # 2. Assert `output` looses to `counterpart`
    equations += [Equation([
        (1, mod.layers[-1][output][-1]),
        (-1, mod.layers[-1][counteroutput][-1]),
    ], Equation.Comparator.GE, epsilon)]

    mod.layers[-1] = [mod.layers[-1][output], mod.layers[-1][counteroutput]]

    outputs = [x[-1] for x in (net.layers[-1]+mod.layers[-1])]

    # Sanity
    nodes = set()
    [nodes.add(node) for l in net.layers for n in l for node in n]
    [nodes.add(node) for l in dup for n in l for node in n]
    for node in nodes: assert nodes.issuperset(node.connectedTo())

    # Return
    inputs = [ns[0] for ns in net.layers[0]]
    return ViewIO(nodes, inputs, outputs, equations)
