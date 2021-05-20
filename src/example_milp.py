import re

def zipa(*lsts):
    assert len(set(map(len, lsts))) == 1
    return zip(*lsts)

# Marabou format changes from time to time! Make sure you get valid results.
def parse_milp(p):
    cont = open(p, "r").read()
    
    try:
        mlayers = []
        for layerid, neuronid, lb, ub in re.findall("Layer ([0-9]+):|\tNeuron([0-9]+)\tLB: ([^,]+), UB: ([^ ]+)", cont):
            if layerid != "":
                assert len(mlayers) == int(layerid)
                mlayers.append([])
            else:
                neuronid = int(neuronid)
                lb = float(lb)
                ub = float(ub)
                mlayers[-1].append((lb, ub))
                assert len(mlayers[-1]) == neuronid

        layers = [mlayers[0]]
        for lb, lf in zip(mlayers[1::2], mlayers[2::2]):
            layers.append([])
            for vb, vf in zipa(lb, lf):
                assert vf[0] >= 0
                # assert abs(max(0, vb[1]) - vf[1]) < 0.01, (vb, vf) <- Bug in Marabou probably
                layers[-1].append( ( vb[0], min(vb[1], vf[1]) ) )
        layers.append(mlayers[-1])
    except:
        print(p)
        raise
    
    return layers

def example_milp():
    # How to obtain MILP bounds for neurons?
    # 1. Apply redy/marabou_patches/milp_bounds.patch on Marabou
    # 2. Run:
    #       ./Marabou --input {NETWORK} --dump-bounds --milp-tightening milp --milp-timeout 10
    #       (change parameters as needed)
    # (if running on a sub-domain, use a Marabou property file to restrict the input or -
    #  create an ipq file using apply_subspace, export_marabou and saveQuery as done in example_basics.py)
    # 3. Parse the output, like the example below

    milp = parse_milp("examples_data/milp_acasxu_5_9.txt")
    print("Neuron 2,4 bounds: %r" % (milp[2][4], ))

if __name__ == "__main__":
    example_milp()

