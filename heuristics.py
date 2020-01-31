'''
Heuristic Algorithms for solving cubic knapsack problems
'''

from cubics import *
from utility import *
from multi_dimensional_knap_lins import *

'''
Apply Glover's constructive Greedy heuristic to a single dimension cubic knapsack
'''
def naive_greedy(cubic):
    n = cubic.n
    c = cubic.c
    C = cubic.C
    D = cubic.D
    a = cubic.a
    b = cubic.b

    # index set for variables x_i currently set to 1
    N1 = set()
    # index set for variables x_i that can feasbily fit
    N0 = set()
    # room left in knapsack
    RHS = b[0]

    # initialize N0 with all item indices whose weight < capacity
    for i in range(n):
        if a[i] < RHS:
            N0.add(i)

    # compute intial value ratios
    # NOTE: initially only the linear contributions are considered
    value_ratios = np.zeros(n)
    for i in N0:
        value_ratios[i] = c[i] / a[i][0]

    # repeatedly pick the best item until no more items can fit
    while RHS > 0 and len(N0) > 0:
        # get the highest value item to include next
        take_index = np.argmax(value_ratios)

        # remove this item from pool of unassigned items
        N0.remove(take_index)
        value_ratios[take_index] = -1

        # update remaining capacity
        RHS = RHS - a[take_index][0]

        # update N0 (remove items which would put us over capacity if taken)
        for i in N0.copy():
            if a[i][0] > RHS:
                N0.remove(i)

        # update value ratios
        for i in N0:
            # add in new quadratic contributions
            value_ratios[i] += (C[i, take_index] + C[take_index, i])/a[i][0]

        # add in new cubic contributions
        for j in N1:
            value_ratios[i] += getDValue(i, j, take_index, D)/a[i][0]

        # add item to taken items
        N1.add(take_index)

    # return selected indices
    return N1


'''
Apply Glover's constructive Greedy heuristic to a single dimension cubic knapsack
'''
def naive_greedy_probe(cubic):
    n = cubic.n
    c = cubic.c
    C = cubic.C
    D = cubic.D
    a = cubic.a
    b = cubic.b

    # index set for variables x_i currently set to 1
    N1 = set()
    # index set for variables x_i that can feasbily fit
    N0 = set()
    # room left in knapsack
    RHS = b[0]

    # initialize N0 with all item indices whose weight < capacity
    for i in range(n):
        if a[i] < RHS:
            N0.add(i)

    # compute intial value ratios
    # NOTE: initially only the linear contributions are considered
    value_ratios = np.zeros(n)
    for i in N0:
        value_ratios[i] = c[i] / a[i][0]

    # repeatedly pick the best item until no more items can fit
    while RHS > 0 and len(N0) > 0:
        # get the 5 best candidate indices
        # TODO make 5 into a variable k
        if len(N0) >= 5:
            candidates = np.argpartition(value_ratios, -5)[-5:]
            #print(candidates)
            combo_evals = {}
            best_eval = 0
            best_combo = -1
            # evaluate each 3-element combination of the cadidates
            for i in range(len(candidates)):
                for j in range(i+1, len(candidates)):
                    for k in range(j+1, len(candidates)):
                        # first make sure this combination doesn't exceed current capacity
                        total_weight = a[i][0] + a[j][0] + a[k][0]
                        if total_weight > RHS:
                            continue
                        combination_profit = 0
                        combination_profit += value_ratios[i]*a[i][0] # CP_i
                        combination_profit += value_ratios[j]*a[j][0] # CP_j
                        combination_profit += value_ratios[k]*a[k][0] # CP_k
                        combination_profit += C[i,j] + C[j,i]
                        combination_profit += C[i,k] + C[k,i]
                        combination_profit += C[j,k] + C[k,j]
                        combination_profit += getDValue(i, j, k, D)
                        # TODO need to add more cubic vals here considering stuff already taken?
                        # TODO sometimes eval is a slightly negative num??
                        combination_profit = combination_profit / total_weight
                        combo_evals[(i,j,k)] = combination_profit
                        if combination_profit > best_eval:
                            best_eval = combination_profit
                            best_combo = (i,j,k)
                        # print(f"({candidates[i]},{candidates[j]},{candidates[k]}) : {combination_profit}")

            # print(combo_evals)
            # print(f"best combo {best_combo}")

            index1_score = 0
            index2_score = 0
            index3_score = 0

            for combo in combo_evals:
                if best_combo[0] in combo:
                    index1_score += combo_evals[combo]
                if best_combo[1] in combo:
                    index2_score += combo_evals[combo]
                if best_combo[2] in combo:
                    index3_score += combo_evals[combo]

            if index1_score > index2_score and index1_score > index3_score:
                take_index = candidates[best_combo[0]]
            elif index2_score > index1_score and index2_score > index3_score:
                take_index = candidates[best_combo[1]]
            else:
                take_index = candidates[best_combo[2]]
        else:
            take_index = np.argmax(value_ratios)

        # remove this item from pool of unassigned items
        N0.remove(take_index)
        value_ratios[take_index] = -1

        # update remaining capacity
        RHS = RHS - a[take_index][0]

        # update N0 (remove items which would put us over capacity if taken)
        for i in N0.copy():
            if a[i][0] > RHS:
                N0.remove(i)

        # update value ratios
        for i in N0:
            # add in new quadratic contributions
            value_ratios[i] += (C[i, take_index] + C[take_index, i])/a[i][0]

        # add in new cubic contributions
        for j in N1:
            value_ratios[i] += getDValue(i, j, take_index, D)/a[i][0]

        # add item to taken items
        N1.add(take_index)

    perform_swaps(N1,N0,cubic)

    # return selected indices
    return N1


'''
Doubly LinkedList class used for efficiently updating N0 as a list of
all untaken items in increasing order of weight
'''
class LinkedList:
    class Node:
        # each node in the LinkedList has a weight and index of the item it represents
        def __init__(self, index, weight):
            # index is the item id (e.g. x1)
            self.index = index
            # weight is a[index]
            self.weight = weight

            # next and prev will be set when node is added to the LinkedList
            self.next = None
            self.prev = None

        # used for printing
        def __str__(self):
            if self.index == -1 and self.weight == -1: return f"(Head)"
            return f"({self.weight},{self.index})"


    # initialize LinkedList with a dummy head Node and size 0
    def __init__(self):
        self.head = self.Node(-1,-1)
        self.size = 0

    # define a generator for iterating over the LinkedList (e.g. 'for item in LinkedList' is now possible)
    def __iter__(self):
        current = self.head.next
        while current is not None:
            yield current
            current = current.next

    # print all of the elements of the list in (index, weight) pairs
    def display(self):
        curr = self.head
        while(curr is not None):
            print(curr, end="")
            curr=curr.next
        print()

    # count the max number of items that can possibly fit and the sum of the weights of those items
    def init_key_index_and_sum(self, RHS):
        curr = self.head.next
        # n0 is the max number of items which can fit
        self.n0 = 0
        # key_sum is the total weight of items 1...n0
        self.key_sum = 0
        # maintain a pointer to the last node of n0
        self.key_pointer = None

        # start by taking lowest weight item, then second lowest, etc.. until no more can fit
        while curr is not None and curr.weight < RHS:
            RHS -= curr.weight
            self.key_sum += curr.weight
            self.n0 += 1
            # TODO more efficient to do this only once after loop finishes loop
            self.key_pointer = curr
            curr = curr.next

    # an item not from the key_set was removed => need to update keyset because it is no logner feasible
    # to take all of the items in it
    def update_key_set(self, RHS):
        while self.key_sum > RHS:
            self.key_sum -= self.key_pointer.weight
            self.n0 -= 1


    # remove the node
    def remove_by_index(self, index):
        curr = self.head
        found = False
        while(curr is not None and not found):
            if curr.index == index:
                found = True
                if curr.next is not None:
                    curr.next.prev = curr.prev
                curr.prev.next = curr.next
                break
            else:
                curr = curr.next
        if not found:
            print('elem not found')
        else:
            self.size -= 1

    # remove all items which don't fit anymore
    # TODO use a Tail and start at the end of list
    def remove_too_big(self, RHS):
        curr = self.head
        count = -1

        # look until we find an item which can not fit
        while (curr is not None and curr.weight < RHS):
            curr = curr.next
            count += 1

        # chopping off an unknown # of items => recount size
        self.size = count

        # all further items are bigger and thus will not fit. So remove them all
        if curr is not None:
            curr.prev.next = None

    # inserts a new node to maintain a sorted list by weight
    def add_node(self, index, weight):
        # create the new node to be inserted
        new_node = self.Node(index, weight)

        # find the spot where it should be added
        curr = self.head
        while (curr.next is not None and curr.next.weight < weight):
            curr = curr.next

        # if not the last element, update the subsequent element to point back correctly
        if curr.next is not None:
            temp = curr.next
            new_node.next = temp
            temp.prev = new_node

        # insert the node
        curr.next = new_node
        new_node.prev = curr

        # increase size by 1
        self.size += 1

    # return the number of elements in the list
    def __len__(self):
        return self.size


'''
Apply Glover's Advanced constructive Greedy heuristic (with product terms) to a single dimension cubic knapsack
'''
def advanced_greedy(cubic, verbose=False):
    n = cubic.n
    c = cubic.c
    C = cubic.C
    D = cubic.D
    a = cubic.a
    b = cubic.b

    # index set for variables x_i currently set to 1
    N1 = set()
    # index set for variables x_i that can feasbily fit
    N0 = LinkedList()
    # room left in knapsack
    RHS = b[0]

    # initialize N0 with all item indices whose weight < capacity
    for i in range(n):
        if a[i] < RHS:
            N0.add_node(i, a[i][0])


    # initialize key_index and key_sum values for N0
    N0.init_key_index_and_sum(RHS)

    if verbose:
        # N0 should be in ascending order by item weight
        print("Initial N0 : ", end="")
        N0.display()
        print(f"Initial RHS: {RHS}")
        print(f"Key Index : {N0.n0} items")
        print(f"Key Index : {N0.key_sum}")

    # intialize value ratios to zero
    value_ratios = np.zeros(n)

    # compute initial value ratios
    for node in N0:
        # compute the max number of copies of x_i which could fit if binary constraint were removed (denoted n_j in the paper)
        max_copies = RHS//node.weight

        if verbose:
            print(f"Max copies : {max_copies}")

        # compute the min of max_copies (n_j) and max_items (n_0)
        n_prime = min(max_copies, N0.n0)

        # initial values deon't consider nonlinear values
        value_ratios[node.index] = n_prime * c[node.index]

    if verbose:
        print(f"Value ratios : {value_ratios}")


    # repeatedly pick the best item until no more items can fit
    while RHS > 0 and len(N0) > 0:
        # get the highest value item to include next
        take_index = np.argmax(value_ratios)

        # if all items have value_ratio of zero. Just take the first item in N0
        # otherwise will pick item 0 even if already picked
        if value_ratios[take_index] == 0:
            take_index = N0.head.next.index

        # remove this item from pool of unassigned items
        N0.remove_by_index(take_index)

        # update remaining capacity
        RHS = RHS - a[take_index][0]

        # Remove items from N0 which don't fit in new capacity
        # TODO need to set value_ratio for removed items to -1
        N0.remove_too_big(RHS)

        # follow fast update procedure from page 8 of Glover's paper on Advanced Greedy Algorithms
        if a[take_index][0] <= N0.key_pointer.weight:
            N0.n0 -= 1
            N0.key_sum -= a[take_index][0]
            if N0.key_pointer.index == take_index:
                N0.key_pointer = N0.key_pointer.prev
        else:
            N0.update_key_set(RHS)

        # recompute value ratios for all untaken items
        # TODO is there a better way than to entirely recompute?
        value_ratios = np.zeros(n)
        for node in N0:
            i = node.index

            # add linear contribution
            value_ratios[i] += c[i]

            for taken_index in N1:
                # add in new quadratic contributions
                value_ratios[i] += (C[i, taken_index] + C[taken_index, i])

                # add in new cubic contributions
                for taken_index2 in N1:
                    if taken_index == taken_index2: continue
                    # TODO counting D val twice for each triplet? Do this without this redundancy
                    value_ratios[i] += getDValue(taken_index, taken_index2, i, D)/2

            # compute the max number of copies of x_i which could fit if binary constraint were removed (denoted n_j in the paper)
            max_copies = RHS//node.weight

            n_prime = min(max_copies, N0.n0)

            value_ratios[i] = n_prime * value_ratios[i]

        # if np.count_nonzero(value_ratios) == 0:
        #     print("No more value to be gained?")
        #     # TODO there still could be more value to gain
        #     break

        # add item to taken items
        N1.add(take_index)

    perform_swaps(N1, N0, cubic)

    # return selected indices
    return N1


def main():
    n = 15
    cdmkp = CMDKP(n=n, density=70, constraints=1)

    # model and solve using the Adams+Forrester linearization
    model1 = standard_lin(cdmkp)
    model1.optimize()
    print('Optimal sol found by linearization : ' + str(model1.objVal))
    for i in range(n):
        print(model1.getVarByName("binary_var["+str(i)+"]"))

    indices = naive_greedy(cdmkp)
    val = getObjVal(indices, cdmkp.c, cdmkp.C, cdmkp.D)
    #print(f"naive_greedy N1: {indices}")
    print(f"naive_greedy obj val: {val}")
    indices = perform_swaps(indices, cdmkp)
    val = getObjVal(indices, cdmkp.c, cdmkp.C, cdmkp.D)
    print(f"new solution: {indices}")
    print(f"new obj val: {val}")


    # a = cdmkp.a
    # total_weight = 0
    # for index in indices:
    #     total_weight += a[index][0]
    #
    # print(f"total weight of sol is {total_weight}")
    # print(f"ttoal knap capacity is {cdmkp.b}")
    # print(f"total weight of sol is {total_weight}")
    # print(f"ttoal knap capacity is {cdmkp.b}")





    # TODO either swaps or getObjVal is wrong
    # sometimes swapping is making things worse

    # indices = naive_greedy_probe(cdmkp)
    # val = getObjVal(indices, cdmkp.c, cdmkp.C, cdmkp.D)
    # #print(f"naive_greedy with probe N1: {indices}")
    # print(f"naive_greedy with probe obj val: {val}")
    #
    # indices = advanced_greedy(cdmkp)
    # val = getObjVal(indices, cdmkp.c, cdmkp.C, cdmkp.D)
    # #print(f"advanced_greedy N1: {indices}")
    # print(f"advanced_greedy obj val: {val}")


if __name__=="__main__":
  main()
