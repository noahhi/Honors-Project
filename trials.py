from cubics import *
from multiple_knap_lins import *
from multiple_heuristics import *

def main():
    setParam('OutputFlag',0)
    setParam('LogFile',"")

    results = {}
    sizes = {10}

    for size in sizes:
        cmkp = CMKP(n=size,density=70,m=3,seed=0)

        # solve exactly using a+f linearization
        model1 = adams_and_forrester_lin(cmkp)
        model1.optimize()
        print('solution found by model1 (standard_lin) : ' + str(model1.objVal))

        print(f"don't force cycles:")

        # solve without forcing cycles (4 diff options)
        for option in range(1,5):
            indices = multiple_knap(cmkp, knap_selection_option=option, force_cycles=False, verbose=False)
            total = get_mult_obj_val(indices, cmkp.c, cmkp.C, cmkp.D)
            print(f"solution found by heuristic w/ option {option} : {total}")


        print(f"force cycles:")

        # solve with forcing cycles (4 diff options)
        for option in range(1,5):
            indices = multiple_knap(cmkp, knap_selection_option=option, force_cycles=True, verbose=False)
            total = get_mult_obj_val(indices, cmkp.c, cmkp.C, cmkp.D)
            print(f"solution found by heuristic w/ option {option} : {total}")


if __name__ == "__main__":
    main()
