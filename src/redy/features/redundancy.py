#
# High-level functions for checking for redundancy
#

from redy.framework.nodes import Node
from redy.framework.equations import Equation
from redy.framework.views import ViewIO
from redy.features import amend, clip

class RedundancyTest(object):
    def __init__(self, network, epsilon):
        # neurons = list of (layer, neuron, mode)
        self.network = network
        self.epsilon = epsilon

    def duplicateAndClip(self, rng, suffix="_dup"):
        net = self.network.duplicate(suffix)

        # Sanities
        assert 0 <= rng.firstLayer < rng.lastLayer < net.layerCount()

        # Clip bdr and fdr
        clip.clipNetwork(net, rng)
        
        return net

    def _prep(self, neurons, rng):
        if rng.lastLayer is None:
            rng = clip.Range(rng.firstLayer, rng.firstMode, self.network.layerCount()-1, rng.lastMode)
        assert all((1 <= l < (self.network.layerCount()-1)) for l,n,f in neurons)
        assert all(( (rng.firstLayer <= l <= rng.lastLayer) and (0 <= n < self.network.layerSize(l)) ) for l, n, f in neurons)

        net = self.duplicateAndClip(rng)
        mod = self.duplicateAndClip(rng, suffix="_mod")
        neurons = {(l-rng.firstLayer, n): f for l, n, f in neurons}
        amend.modify(mod, neurons)
        return net, mod, neurons

    # Given a list of neurons (and optionally clipping range) returns
    # a modified network
    # `rng` - optional clipping range, see examples
    def getModified(self, neurons, rng=clip.Range(), returnNetwork=False):
        net, mod, neurons = self._prep(neurons, rng)
        if not returnNetwork: return mod
        else: return (net, mod)

    # Given a list of neurons (and optionally clipping range) returns
    # a ViewIO which compares between the original network and the modified one.
    # `neruons` - a list of neurons to modify
    # `comparator` - how should the outputs be compared? (gt or lt)
    # `output` - output number to compare
    # `rng` - optional clipping range, see examples
    # `returnNetwork` - allows to return NetworkViews of the networks joined in the ViewIO (`view`)
    def getComparedExact(self, neurons, comparator, output, rng=clip.Range(), returnNetwork=False):
        net, mod, neurons = self._prep(neurons, rng)
        view = amend.compareExact(net, mod, neurons, comparator, output, self.epsilon)
        if not returnNetwork: return view
        else: return (view, net, mod)

    # Given a list of neurons (and optionally clipping range) returns
    # a ViewIO which compares between the original network and the modified one
    # (result-preserving wise, minimum class wins)
    # `neruons` - a list of neurons to modify
    # `output` - output number to be compared
    # `counteroutput` - output to compare with
    # `rng` - optional clipping range, see examples
    # `returnNetwork` - allows to return NetworkViews of the networks joined in the ViewIO (`view`)
    # This function generates a query which looks for inputs for which
    # `output` is the winning class in the original, but `counteroutput` wins
    # `output` in the modified network.
    def getComparedMinimum(self, neurons, output, counteroutput, rng=clip.Range(), returnNetwork=False):
        assert (rng.lastLayer, rng.lastMode) in [(None, 0), (self.network.layerCount()-1, 0)]
        net, mod, neurons = self._prep(neurons, rng)
        view = amend.compareMinimum(net, mod, neurons, output, counteroutput, self.epsilon)
        if not returnNetwork: return view
        else: return (view, net, mod)

    # Given a list of neurons (and optionally clipping range) returns
    # a ViewIO with the original and modified networks are joined (without any comparison)
    # `neruons` - a list of neurons to modify
    # `output` - output number to be compared
    # `counteroutput` - output to compare with
    # `rng` - optional clipping range, see examples
    # `returnNetwork` - allows to return NetworkViews of the networks joined in the ViewIO (`view`)
    def getJoined(self, neurons, rng=clip.Range(), returnNetwork=False):
        net, mod, neurons = self._prep(neurons, rng)
        view = amend.join(net, mod, neurons)
        if not returnNetwork: return view
        else: return (view, net, mod)

    # Given a neuron, returns a query (ViewIO) which checks if the neuron is
    # not in the given state (see examples).
    # `neruon` - (layer, neuron, function)
    # `rng` - optional clipping range, see examples
    # `returnNetwork` - allows to return NetworkViews of the networks joined in the ViewIO (`view`)
    # `strict` - if true, checks in a way with 0 False-Positives (x >= -e or x <= e)
    #            otherwise, checks in a way with 0 False-Negatives (x >= e or x <= -e)
    def getStateCheck(self, neuron, rng=clip.Range(), returnNetwork=False, strict=False):
        assert (rng.lastLayer, rng.lastMode) in [(None, 0), (self.network.layerCount()-1, 0)]

        l,n,f = neuron
        rng = clip.Range(firstLayer=rng.firstLayer, firstMode=rng.firstMode, lastLayer=l, lastMode=0)
        l -= rng.firstLayer

        clipped = self.duplicateAndClip(rng)
        clipped.layers[-1] = clipped.layers[-1][n:n+1]

        assert len(clipped.layers[-1][0]) == 1
        vb = clipped.layers[-1][0][0]

        if not strict:
            comp, eps = (Equation.Comparator.GE, self.epsilon) if f == "inactive" else (Equation.Comparator.LE, -self.epsilon)
        else:
            comp, eps = (Equation.Comparator.GE, -self.epsilon) if f == "inactive" else (Equation.Comparator.LE, self.epsilon)
        equations = [Equation([
            (1, vb)
        ], comp, eps)]

        # Return
        inputs = [ns[0] for ns in clipped.layers[0]]
        view = ViewIO(clipped.nodes(), inputs, [vb], equations)

        if returnNetwork: return view, clipped
        else: return view

