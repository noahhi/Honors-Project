from cubics import *
from multi_dimensional_knap_lins import *
from heuristics import *
from timeit import default_timer as timer
import pandas as pd

def main():
    setParam('OutputFlag',0)
    setParam('LogFile',"")

    # store results of trials
    data = []
    sizes = [35,40,45,50]
    dimensions = [3]
    seed = 0

    for size in sizes:
        for dimension in dimensions:
            print(f"size is {size} with {dimension} dimensions")
            cdmkp = CMDKP(n=size,density=70,constraints=dimension,seed=seed)

            # solve exactly using a+f linearization
            # start = timer()
            # model1 = standard_lin(cdmkp)
            # model1.optimize()
            # time_af = timer() - start
            # print(f'solution found by model1 (standard_lin) : {model1.objVal:.2f} took {time_af:.2f} seconds')

            start = timer()
            model1 = naive_greedy(cdmkp)
            time_naive = timer() - start
            obj_naive = getObjVal(model1, cdmkp.c, cdmkp.C, cdmkp.D)

            # start = timer()
            # model2 = naive_greedy_probe(cdmkp)
            # time_probe = timer() - start
            # obj_probe = getObjVal(model2, cdmkp.c, cdmkp.C, cdmkp.D)

            # start = timer()
            # model3 = advanced_greedy(cdmkp)
            # time_advanced = timer() - start
            # obj_advanced = getObjVal(model3, cdmkp.c, cdmkp.C, cdmkp.D)

            res_dict = {"size":size, "dimensions":dimension, "seed":seed, "naive_obj":obj_naive}
                                                            #"advanced_obj":obj_naive}
            data.append(res_dict)

        # save results after each size
        print("saving...")
        df = pd.DataFrame(data)
        df.to_pickle('plots/multidimensional_compare.pkl')

    print("final df")
    print(df)


if __name__ == "__main__":
    main()
