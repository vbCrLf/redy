#
# Convert from Redy representation (ViewIO) into
# an object for easy network evaluation.
# See `export_evaluate`
#

from redy.framework import nodes
from redy.framework import equations

EPSILON = 1e-13

# Converts a ViewIO into an object for evaluation
def export_evaluate(view):
    ev = Evaluator()

    nl = list(view.nodes)
    
    trans = nl.index
    ev.numVars = len(nl)
    ev.inputVars = [trans(n) for n in view.inputs]
    ev.outputVars = [trans(n) for n in view.outputs]

    for node in nl:
        nodev = trans(node)
        lower, upper = node.limit
        if lower is not None: ev.lowerBounds[nodev] = lower
        if upper is not None: ev.upperBounds[nodev] = upper

        if isinstance(node, nodes.NodeSum):
            terms = []
            for c, v in node.inputs:
                terms.append((trans(v),c))
            terms.append( (nodev, -1) )
            ev.equList.append( (0, -node.scalar, terms) )
        elif isinstance(node, nodes.NodeReLU):
            ev.constraints.append( ("relu", trans(node.input), nodev) )
        elif type(node) == nodes.Node:
            pass
        else:
            assert False, "Unknown node type %r" % (node, )

    compareTrans = {
        equations.Equation.Comparator.EQ: 0,
        equations.Equation.Comparator.GE: 1,
        equations.Equation.Comparator.LE: 2,
    }
    for equation in view.equations:
        terms = []
        for c, v in equation.terms:
            terms.append( (trans(v), c) )
        ev.equList.append( (compareTrans[equation.comparator], equation.scalar, terms) )

    for n in nl:
        ev.translate[n] = trans(n)
        ev.translate[trans(n)] = n

    return ev

class Evaluator(object):
    def __init__(self):
        self.translate = {}
        self.numVars = 0
        self.inputVars = []
        self.outputVars = []

        self.lowerBounds = {}
        self.upperBounds = {}

        self.equList = [] # List of (equType, scalar, adds). equType = enum(equ, ge, le)
        # TODO: Support other types, like abs
        self.constraints = [] # List of ("relu", b, f)

    # Evaluate the network with input `inp`.
    # `validate` - should equations be validated after evaluation?
    #              note that forward evaluation may fail, and without validation
    #              it will go unnoticed.
    # `returnValidation` - return validation result instead of throwing AssertionFailed
    def forwardEvaluate(self, inp, validate=True, returnValidation=False):
        d = {}

        assert len(inp) == len(self.inputVars), "Unexpected input size"

        # Set input
        for v, c in zip(self.inputVars, inp):
            d[v] = c

        # Set all fixed variables
        for v,l in self.lowerBounds.items():
            if v not in self.upperBounds: continue
            u = self.upperBounds[v]
            if l == u: d[v] = u

        # Forward evaluate
        while True:
            cont = False

            for equType, scalar, adds in self.equList:
                if equType != 0: continue
                newVars = [(v, c) for v, c in adds if v not in d]
                if len(newVars) != 1: continue
                nv, nc = newVars[0]

                d[nv] = -(sum(d[v]*c for v, c in adds if v != nv) - scalar) / nc
                cont = True
            if cont: continue

            for t, vb, vf in self.constraints:
                assert t == "relu", "Only ReLU is currently supported"
                if vb in d and vf not in d:
                    d[vf] = max(0, d[vb])
                    cont = True
            if cont: continue

            break

        # Validate if required
        if validate:
            valid = False
            try:
                self.validate(d)
                valid = True
            except AssertionError:
                if not returnValidation: raise
            if returnValidation:
                return valid, d

        # Return
        return d
    
    def validate(self, d):
        # Did all variables got assignments?
        assert len(d) == self.numVars

        # Are all variables are in bound?
        for v, c in self.lowerBounds.items(): assert d[v] >= c - EPSILON, v
        for v, c in self.upperBounds.items(): assert d[v] <= c + EPSILON, v

        # Are there any violated equations?
        for i, (equType, scalar, adds) in enumerate(self.equList):
            res = sum(d[v]*c for v, c in adds) - scalar

            if equType == 0:
                assert abs(res) < EPSILON
            elif equType == 1:
                assert res >= -EPSILON
            elif equType == 2:
                assert res <= EPSILON
            else:
                assert False, equType

        # Are there any violated piecewise-linear constraints?
        for t, vb, vf in self.constraints:
            assert t == "relu", "Only ReLU is currently supported"
            assert d[vf] == max(0, d[vb])

        # Good!
