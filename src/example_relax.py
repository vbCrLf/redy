from maraboupy import Marabou

from redy.features import redundancy
from redy.convert import import_nnet, export_evaluate
from redy.framework.equations import Equation

ACAS5_9 = "./examples_data/acasxu/ACASXU_experimental_v2a_5_9.nnet"

def example_relax():
    net = import_nnet.import_nnet(open(ACAS5_9, "r").read())
    redtest = redundancy.RedundancyTest(net, 1e-4)

    # Convert neuron 2,34 into a relaxed l_m function,
    # given lower and upper bounds
    li, ni = (2, 4)
    lower, upper = -7.5987, 1.5147 # In a real scenario, use an output from
                                   # MILP, see MILP example.
    net = redtest.getModified([(li, ni, "nofunc")])
    io = net.toViewIO()

    vb = net.layers[li][ni][0]
    vf = net.layers[li][ni][1]

    # vf - (u / (u-l)) * vb = (-l*u) / (2*(u-l)) -- See paper
    lb,ub = lower,upper
    io.equations.append(Equation([
        (1, vf),
        (-(ub / (ub-lb)), vb)
    ], Equation.Comparator.EQ, (-lb*ub) / (2*(ub-lb))))

    # Evaluate the modified network
    e = export_evaluate.export_evaluate(io)
    print(e.forwardEvaluate([0.12]*5))

    # Lambdas for calculating l_m and error values
    l_m_error = lambda ml, mu: ((-ml*mu) / (2.*(mu-ml)))
    l_m_eval = lambda ml, mu, x: (mu / (mu - ml))*x + l_m_error(ml, mu)

if __name__ == "__main__":
    example_relax()

