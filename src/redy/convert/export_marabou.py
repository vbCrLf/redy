#
# Convert from Redy representation (ViewIO) into
# Marabou object for query dispatching.
# See `export_marabou`
#

import numpy as np

from redy.framework import nodes
from redy.framework import equations

from maraboupy.MarabouNetwork import MarabouNetwork
from maraboupy import MarabouUtils
from maraboupy import MarabouCore

def reachable_from_input(view):
    reachable = []
    current = view.inputs
    while len(current) > 0:
        new_current = []

        for n in current: _reachable_from_input(view, reachable, n)
        for eq in view.equations:
            if eq.comparator != equations.Equation.Comparator.EQ: continue

            terms = set([n for _, n in eq.terms])
            not_reached = [n for n in terms if n not in reachable]
            if len(not_reached) == 1: new_current.append(not_reached[0])

        current = new_current

    return reachable

def _reachable_from_input(view, reachable, node):
    if node in reachable: return
    reachable.append(node)
    for n in get_node_next(view, node): _reachable_from_input(view, reachable, n)

def get_node_next(view, node):
    outputs = []
    for n in view.nodes:
        if node in n.connectedTo(): outputs.append(n)
    return outputs

# The Marabou version we used had a bug where variables
# unreachable from the input would get discarded (even if they
# may affect other neurons). This adds them to the input to
# mitigate the bug.
def mitigate_marabou_constant_nodes_bug(view, mnn):
    #suspected = [node for node in view.nodes if type(node) == nodes.Node and node not in view.inputs]
    reachable = set(reachable_from_input(view))

    suspected = set(view.nodes) - reachable
    assert all(node.limit[0] == node.limit[1] for node in suspected)

    old_input = list(mnn.inputVars[0].tolist())
    to_be_added = [mnn.translate[n] for n in suspected]
    assert all([(n not in old_input) for n in to_be_added])

    new_input = list(old_input)
    new_input.extend(to_be_added)
    mnn.inputVars = [np.array(new_input)]

    mnn.realInputs = old_input

# Convert ViewIO to a Marabou object
# Note: this method add nodes to the input. (see call to `mitigate_marabou_constant_nodes_bug` below)
def export_marabou(view):
    nn = MarabouNetwork()

    # Important note: Marabou does not preserve input order, so their's
    # variables must be created in the required order
    trans = {n: nn.getNewVariable() for n in view.inputs + list(set(view.nodes) - set(view.inputs))}
    assert len(trans) == len(view.nodes)

    # Convert every Node to Marabou representation
    for node in view.nodes:
        nodev = trans[node]
        lower, upper = node.limit
        if lower is not None: nn.setLowerBound(nodev, lower)
        if upper is not None: nn.setUpperBound(nodev, upper)

        if isinstance(node, nodes.NodeSum):
            e = MarabouUtils.Equation()
            for c, v in node.inputs:
                e.addAddend(c, trans[v])
            e.addAddend(-1, nodev)
            e.setScalar(-node.scalar)
            nn.addEquation(e)
        elif isinstance(node, nodes.NodeReLU):
            bf = (trans[node.input], nodev)
            nn.addRelu(*bf)
            if node.relaxed:
                nn.relaxedReluList.append(bf)
        elif isinstance(node, nodes.NodeAbs):
            bf = (trans[node.input], nodev)
            nn.addAbsConstraint(*bf)
        elif type(node) == nodes.Node:
            pass
        else:
            assert False, "Unknown node type %r" % (node, )

    # Convert equations
    compareTrans = {
        equations.Equation.Comparator.GE: MarabouCore.Equation.GE,
        equations.Equation.Comparator.LE: MarabouCore.Equation.LE,
        equations.Equation.Comparator.EQ: MarabouCore.Equation.EQ,
    }
    for equation in view.equations:
        e = MarabouUtils.Equation(compareTrans[equation.comparator])
        for c, v in equation.terms:
            e.addAddend(c, trans[v])
        e.setScalar(equation.scalar)
        nn.addEquation(e)

    # Assign inputs/outputs
    nn.inputVars = [np.array([trans[n] for n in view.inputs])]
    nn.outputVars = np.array([trans[n] for n in view.outputs])

    # Table for future conversion from ViewIO Nodes to Marabou variables and conversely
    nn.translate = {}
    for n,v in trans.items():
        nn.translate[n] = v
        nn.translate[v] = n

    # Note: this method add nodes to the input. Use nn.realInputs
    mitigate_marabou_constant_nodes_bug(view, nn)

    return nn

