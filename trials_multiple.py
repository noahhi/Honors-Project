from cubics import *
from multiple_knap_lins import *
from multiple_heuristics import *
from timeit import default_timer as timer
import pandas as pd

def main():
    setParam('OutputFlag',0)
    setParam('LogFile',"")

    # store results of trials
    data = []
    sizes = [100,125,150,175,200,225,250]
    ms = [3]
    dimensions = []

    for size in sizes:
        for m in ms:
            print(f"size is {size} with m={m}")
            cmkp = CMKP(n=size,density=70,m=m,seed=0)

            # solve exactly using a+f linearization
            # start = timer()
            # model1 = standard_lin(cmkp)
            # model1.optimize()
            # time_af = timer() - start
            # print(f'solution found by model1 (standard_lin) : {model1.objVal:.2f} took {time_af:.2f} seconds')

            res_dict = {"size":size, "m":m, "seed":0}

            # solve without forcing cycles (4 diff options)
            for option in range(1,5):
                start = timer()
                indices = multiple_knap(cmkp, knap_selection_option=option, force_cycles=False, verbose=False)
                total = get_mult_obj_val(indices, cmkp.c, cmkp.C, cmkp.D)
                end = timer()
                time = end - start
                print(f"solution found by heuristic w/ option {option} : {total}. took {time} seconds")
                res_dict[f"option{option}"] = total
                res_dict[f"option{option}:time"] = time


            # print(f"force cycles:")
            #
            # # solve with forcing cycles (4 diff options)
            # for option in range(1,5):
            #     indices = multiple_knap(cmkp, knap_selection_option=option, force_cycles=True, verbose=False)
            #     total = get_mult_obj_val(indices, cmkp.c, cmkp.C, cmkp.D)
            #     print(f"solution found by heuristic w/ option {option} : {total}")


            data.append(res_dict)

        # save results after each size
        print("saving...")
        df = pd.DataFrame(data)
        df.to_pickle('heuristic_results_multiple_no_cycles_big.pkl')
        df.to_excel('heuristic_results_multiple_no_cycles_big.xlsx')

    print("final df")
    print(df)


if __name__ == "__main__":
    main()
