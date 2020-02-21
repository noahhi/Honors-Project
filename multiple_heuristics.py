from cubics import *
from multiple_knap_lins import *
from utility import *
import numpy as np
import math


# Get objVal of a multiple knapsack solution
def get_mult_obj_val(N1, c, C, D):
    total = 0

    # add up contribution of each knapsack
    for knap in N1:
        total += getObjVal(N1[knap], c, C, D)

    return total


'''
Heuristic for a cubic multiple knapsack problem instance which selects the knapsack to be added to
at each iteration based on the specified selection criterion

Params:
    knap_selection_options: a number, 1-4, which specifies which criterion to use
    force_cycles: Specifies whether or not to enfore cyclical knapsack selection
'''
def multiple_knap(cubic, knap_selection_option=4, force_cycles=True, oscillation_levels=[0.25,0.5,0.75,1.0],
                    verbose=True, hovering=False):
    n = cubic.n
    c = cubic.c
    C = cubic.C
    D = cubic.D
    a = cubic.a
    b = cubic.b
    m = cubic.m


    # index set for variables x_i currently set to 1
    # 1 set for each knapsack to store items in that knapsack
    N1 = dict()
    for k in range(m):
        N1[k] = set()

    # set of knapsacks which still have room for more items
    activeK = set()
    for k in range(m):
        activeK.add(k)

    # init room remaining in each knapsack
    RHS = np.zeros(m)
    for i in range(m):
        RHS[i] = b[i]

    # initialize NK for each knapsack with all item indices whose weight < capacity
    # Nk[k] is the set of indices which can still fit in knapsack k (analagous to N0)
    N0 = {}
    for k in activeK:
        N0[k] = set()
        for i in range(n):
            if a[i] < RHS[k]:
                N0[k].add(i)

    if force_cycles:
        # keep track of remaining active knapsacks in the current cycle
        temp_active = activeK.copy()

    # initialize hover levels for each knapsack
    if hovering:
        hover_levels = np.zeroes(m)

    while len(activeK) > 0:

        # apply knapsack choice selection rule for selecting knapsack k*
        knapsack_choice = -1

        if force_cycles:
            if len(temp_active) <= 0:
                # finished a cycle. Reset choices
                temp_active = activeK.copy()
            pool = temp_active
        else:
            pool = activeK

        # choose knapsack with lowest remaining capacity
        if knap_selection_option == 1:
            minRHS = math.inf
            for k in pool:
                if RHS[k] < minRHS:
                    minRHS = RHS[k]
                    knapsack_choice = k

        # choose knapsack with the largest remaining capacity
        elif knap_selection_option == 2:
            maxRHS = -math.inf
            for k in pool:
                if RHS[k] > maxRHS:
                    maxRHS = RHS[k]
                    knapsack_choice = k

        # choose knapsack with biggest possible addition
        elif knap_selection_option == 3:
            max_val = -math.inf
            for k in pool:
                (best_item, best_value) = get_item_naive(cubic, N0=N0[k], N1=N1[k])
                if best_value > max_val:
                    max_val = best_value
                    knapsack_choice = k

        # choose knapsack with biggest difference between its best and second best items
        elif knap_selection_option == 4:
            max_diff = -math.inf
            for k in pool:
                diff = get_biggest_diff(cubic, N0=N0[k], N1=N1[k])
                if diff > max_diff:
                    max_diff = diff
                    #print(f'new max diff is {max_diff} : knapsack {k}')
                    knapsack_choice = k

        else:
            raise ValueError(f"{knap_selection_option} is not a valid knapsack selection option")

        if force_cycles:
            # prevent picking this knap again until a full cycle is done
            temp_active.remove(knapsack_choice)

        if verbose:
            print(f"k* by option {knap_selection_option} is {knapsack_choice}")


        # apply combination evaluation method to knpasack k*
        best_item = get_item_naive(cubic, N0=N0[k], N1=N1[k])[0]

        # update N1 and RHS for choosen knapsack
        N1[knapsack_choice].add(best_item)
        RHS[knapsack_choice] -= a[best_item]


        # update N0 (remove items which would put us over capacity if taken)
        for i in N0[knapsack_choice].copy():
            if a[i] > RHS[knapsack_choice]:
                N0[knapsack_choice].remove(i)

        # remove item from N0 pool for all knapsacks
        for k in activeK.copy():
            if best_item in N0[k]:
                N0[k].remove(best_item)

            # remove knapsack from activeK if no more items can be fit
            if len(N0[k]) == 0:
                activeK.remove(k)
                if force_cycles:
                    if k in temp_active:
                        temp_active.remove(k)

        ### Hover Oscillation ###
        if hovering:
            # calculate how full the knapsack is
            percent_full = (b[m]-RHS[knapsack_choice])/b[m]
            print(f"knapsack {knapsack_choice} is currently at {percent_full}% capacity")

            # check if we surpassed the current hover level, if we have, perform local improvement
            curr_hover_level = hover_levels[knapsack_choice]
            if curr_hover_level < len(oscillation_levels) and percent_full > oscillation_levels[curr_hover_level]:
                # perform local improvement for some number of iterations
                perform_swaps(N1[knapsack_choice], cubic, N0=N0[knapsack_choice],RHS=RHS[knapsack_choice])

                #NOTE need to return all removed and added items so we can update N1 and N0

                # make sure to update N0 for each knapsack (may need to add and/or remove items)


                # increment the hover level for this knapsack
                hover_levels[knapsack_choice]+= 1


    return N1


'''
Retrieve and return the highest value item given a single knapsack.

Params:
    N0 : set of items to be considered
    N1 : set of items already in knapsack
'''
def get_item_naive(cubic, N0, N1):
    c = cubic.c
    C = cubic.C
    D = cubic.D
    a = cubic.a

    N1 = list(N1)

    value_ratios = {}
    for i in N0:
        value_ratios[i] = c[i] / a[i]

        for j in range(len(N1)):
            value_ratios[i] += C[i,N1[j]] + C[N1[j],i]
            for k in range(j+1, len(N1)):
                value_ratios[i] += getDValue(i,N1[j],N1[k],D)

    best_item = max(value_ratios, key=lambda key:value_ratios[key])

    return (best_item, value_ratios[best_item])

'''
Utility function for knapsack selection option 4
Returns the knapsack with the biggest diff between the 1st and 2nd most valuable items

Params:
    N0 : set of items to be considered
    N1 : set of items already in knapsack
'''
def get_biggest_diff(cubic, N0, N1):
    c = cubic.c
    C = cubic.C
    D = cubic.D
    a = cubic.a

    N1 = list(N1)

    value_ratios = {}
    for i in N0:
        value_ratios[i] = c[i] / a[i]

        for j in range(len(N1)):
            value_ratios[i] += C[i,N1[j]] + C[N1[j],i]
            for k in range(j+1, len(N1)):
                value_ratios[i] += getDValue(i,N1[j],N1[k],D)

    # TODO if only 1 item maybe should prioritize this one more?? --> return some high value?
    if len(value_ratios) == 1:
        # if only 1 item, return its value
        return list(value_ratios.values())[0]

    #TODO sometimes value_Ratios is empty --> error. Why is this occuring?
    # best value
    best_val = list(sorted(value_ratios.values()))[-1]

    # second best val
    second_best_val = list(sorted(value_ratios.values()))[-2]

    return best_val - second_best_val



def main():
    n = 12
    cmkp = CMKP(n=n, density=70, m=4)

    model1 = standard_lin(cmkp)
    model1.optimize()
    print('solution found by model1 (standard_lin) : ' + str(model1.objVal))

    indices = multiple_knap(cmkp, knap_selection_option=3, verbose=False, force_cycles=True)
    total = get_mult_obj_val(indices, cmkp.c, cmkp.C, cmkp.D)
    print(f"objective value is {total}")

    indices = multiple_knap(cmkp, knap_selection_option=3, verbose=False, force_cycles=False)
    total = get_mult_obj_val(indices, cmkp.c, cmkp.C, cmkp.D)
    print(f"objective value is {total}")


if __name__=="__main__":
  main()
