from maraboupy import Marabou, MarabouCore

from redy.features import redundancy, clip
from redy.convert import import_nnet, export_marabou, export_evaluate

ACAS5_9 = "./examples_data/acasxu/ACASXU_experimental_v2a_5_9.nnet"

SPLIT = 2
def apply_subspace(io, subspace):
    assert type(subspace) is str
    assert len(subspace) % len(io.inputs) == 0

    subspace = list(map(int, subspace))
    for split in range(0, len(subspace), len(io.inputs)):
        split = subspace[split:split+len(io.inputs)]

        for inp, ch in zip(io.inputs, split):
            assert ch < SPLIT
            l,u = inp.limit
            sz = u-l
            chsz = sz/float(SPLIT)
            inp.updateLimit(l+ch*chsz,l+(ch+1)*chsz)

def example_basics():
    net = import_nnet.import_nnet(open(ACAS5_9, "r").read())
    redtest = redundancy.RedundancyTest(net, 1e-4)

    ns = [(2, 34, "active"), (2, 39, "inactive"), (3, 0, "inactive")]

    # Get a new network with `ns` neurons removed (inactive => replace neuron with 0,
    #                                              active => replace neuron with identity
    #                                              nofunc => remove any connection between backward and forward value
    #                                                        [used for relax, see other examples])
    mod0 = redtest.getModified(ns)

    # Get network with layers after Layer 3 clipped
    mod4 = redtest.getModified(ns, clip.Range(lastLayer=3, lastMode=1))

    # Check if outout #1 in the original is greater than in the modified
    com0 = redtest.getComparedExact(ns, "gt", 1)
    
    # Check if outout #2 in the original is greater than in the modified,
    # but with layers after Layer 4 clipped (e.g. k-forward-redundancy)
    com2 = redtest.getComparedExact(ns, "lt", 2, clip.Range(lastLayer=4, lastMode=0))
    
    # Check if output #0 wins in the original network, but loses to output #1
    # in the modified network (e.g. result-preserving redundancy)
    min1 = redtest.getComparedMinimum(ns, 0, 1)

    # Check if neuron (2, 34) is active (by finding a counter-example) (e.g. phase-redundancy)
    sum0 = redtest.getStateCheck(ns[0])

    # Evaluate a modified network
    e = export_evaluate.export_evaluate(mod4.toViewIO())
    print(e.forwardEvaluate([0.12]*5))

    # Optionally restrict the network into a subdomain.
    # Subdomain is represented in a list of digits. In the case of 5 inputs,
    # each 5 digits corresponds to a split where each digit corresponds to a coordinate,
    # instructing which range of the coordinate to take.
    # In this example (00000 10101), two splits are done for each coordinate -
    #  - in the first split, the first half of each coordinate is taken (00000)
    #  - in the second split:
    #       - second half of #1 coordinate (1), first half of #2 coordinate (0),
    #         second half of #3 coordinate (1), ...
    apply_subspace(com2, "0000010101")

    # Run a Marabou query
    mara = export_marabou.export_marabou(com2)
    ipq = mara.getMarabouQuery() # Marabou query
    # MarabouCore.saveQuery(ipq, "query.ipq") # If the query is needed for later or external query solving
    print(assert_no_to(mara.solve()))

    # Use SNC for faster queries and/or verbosity:
    #mara.solve(options=Marabou.createOptions(verbosity=3, snc=True, numWorkers=8))

def assert_no_to(v):
    vals, stats = v
    assert not stats.hasTimedOut()
    return vals

if __name__ == "__main__":
    example_basics()

