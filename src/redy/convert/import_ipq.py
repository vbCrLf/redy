#
# Convert from a Marabou InputQuery file (.ipq) to a ViewNetwork
# See `import_ipq`
#

from itertools import chain

from redy.framework import nodes, views, equations

NNET_COUNTER = 0

def build_weighted_sum(q, var, l, netPrefix, strict):
    layer = []
    dvar = {}
    ni = 0
    for equType, scalar, adds in q.equList:
        if strict:
            assert equType == 0
        else:
            if equType != 0: continue
        newVars = [(v, c) for v, c in adds if v not in var]
        if strict: assert len(newVars) in [0, 1, len(adds)]
        if len(newVars) != 1: continue
        nv,nc = newVars[0]

        if strict:
            assert nc == -1.0
        else:
            assert nc in [-1.0, 1.0] 

        n = nodes.NodeSum()
        n.name = "{}{:02}_w{:02}".format(netPrefix, l, ni)
        n.scalar = -scalar # TODO: i think so
        n.inputs = [(c, var[v]) for v,c in adds if v != nv]
        layer.append(n)
        assert nv not in dvar
        dvar[nv] = n

        ni += 1

    var.update(dvar)

    return layer

def build_relu(q, var, l, netPrefix, strict):
    layer = []
    dvar = {}
    ni = 0

    for tt, vb, vf in q.constraints:
        assert tt == "relu"
        if vb not in var or vf in var: continue

        n = nodes.NodeReLU()
        n.name = "{}{:02}_r{:02}".format(netPrefix, l, ni)
        n.input = var[vb]
        layer.append(n)
        assert vf not in dvar
        dvar[vf] = n

        ni += 1

    var.update(dvar)

    return layer

# Converts an ipq file (contents) into a ViewNetwork.
# Note: For now, assumes a strict network structure
def import_ipq(ipq, strict=True):
    global NNET_COUNTER

    netPrefix = "ipq{}_".format(NNET_COUNTER)
    NNET_COUNTER += 1

    q = Query(ipq)

    # Mitigate Marabou bug. See `mitigate_marabou_constant_nodes_bug` in `export_marabou.py` for more information
    original_inputs = list(q.inputVars)
    fixed_vars = [v for v in range(q.numVars) if v in q.lowerBounds and v in q.upperBounds and q.lowerBounds[v] == q.upperBounds[v]]
    q.inputVars += fixed_vars

    # Create all input Nodes
    var = {}
    layers = [[]]
    for i, v in enumerate(q.inputVars):
        n = nodes.Node()
        n.name = "{}{:02}_{:02}".format(netPrefix, 0, i)
        layers[-1].append(n)
        var[v] = n
    
    # Iteratively build WS and ReLU layers
    while True:
        layer = build_weighted_sum(q, var, len(layers), netPrefix, strict)
        assert layer != []
        layers += [layer]
        
        layer = build_relu(q, var, len(layers), netPrefix, strict)
        if layer == []: break
        layers += [layer]

    # Make sure all outputs are in the last layer
    if strict:
        assert set([var[v] for v in q.outputVars]) == set(layers[-1])
        # assert len(var) == q.numVars # comment this out, there may be unnecessary variables

    # Make sure all ReLUs are in the new representation
    assert all([(c[0] != "relu" or (c[1] in var and c[2] in var)) for c in q.constraints])
    for equType, scalar, adds in q.equList:
        assert all((v[0] in var) for v in adds), (equType, scalar, adds)

    # Copy neurons bounds
    for v in var:
        l = q.lowerBounds.get(v, None)
        u = q.upperBounds.get(v, None)
        var[v].updateLimit(l, u)

    # Convert to layers with neurons, where each neuron is (WS, ReLU)
    combined = []
    combined.append([[x] for x in layers[0]])
    for i in range(1, len(layers)-1, 2):
        pairs = []
        used = []
        for v in layers[i+1]:
            assert v.input in layers[i]
            pairs.append([v.input, v])
            used.append(v)
            used.append(v.input)

        for v in layers[i] + layers[i+1]:
            if v not in used: pairs.append([v])

        combined.append(pairs)
    combined.append([[x] for x in layers[-1]])

    # Create the object
    vn = views.ViewNetwork(combined)
    vn.originalInputs = original_inputs
    return vn

class Query(object):
    def __init__(self, s):
        s = s.splitlines()

        read = lambda ll, f: (ll[1:], f(ll[0]))
        s, self.numVars = read(s, int)
        s, lowerBoundsCount = read(s, int)
        s, upperBoundsCount = read(s, int)
        s, equationsCount = read(s, int)
        s, plConstraintsCount = read(s, int)

        s, inputVarsCount = read(s, int)
        self.inputVars = []
        for i in range(inputVarsCount):
            s, cur = read(s, str)
            cur = cur.split(",")
            assert len(cur) == 2 and int(cur[0]) == i
            var = int(cur[1])
            assert var < self.numVars
            self.inputVars.append(var)

        s, outputVarsCount = read(s, int)
        self.outputVars = []
        for i in range(outputVarsCount):
            s, cur = read(s, str)
            cur = cur.split(",")
            assert len(cur) == 2 and int(cur[0]) == i
            var = int(cur[1])
            assert var < self.numVars
            assert var not in self.inputVars
            self.outputVars.append(var)

        self.lowerBounds = {}
        self.upperBounds = {}
        for i, d in chain(enumerate([self.lowerBounds]*lowerBoundsCount),
                          enumerate([self.upperBounds]*upperBoundsCount)):
            s, cur = read(s, str)
            cur = cur.split(",")
            assert len(cur) == 2
            v, c = int(cur[0]), float(cur[1])
            assert v not in d
            d[v] = c

        self.equList = []
        for i in range(equationsCount):
            s, cur = read(s, str)
            cur = cur.split(",")
            assert int(cur[0]) == i
            equType = int(cur[1])
            scalar = float(cur[2])
            adds = []
            for v, c in zip(cur[3::2], cur[4::2]):
                adds.append((int(v), float(c)))
            self.equList.append((equType, scalar, adds))

        self.constraints = []
        for i in range(plConstraintsCount):
            s, cur = read(s, str)
            cur = cur.split(",")
            assert int(cur[0]) == i
            assert cur[1] in ["relu", "absoluteValue", "max"]

            if cur[1] in ["relu", "absoluteValue"]:
                vf = int(cur[2])
                vb = int(cur[3])
                self.constraints.append((cur[1], vb, vf))
            else:
                assert cur[1] == "max"
                vf = int(cur[2])
                vb = [int(x) for x in cur[3:]]
                self.constraints.append((cur[1], vb, vf))
