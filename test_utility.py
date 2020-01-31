import unittest

from utility import perform_swaps
from cubics import *
from heuristics import *

class TestSwaps(unittest.TestCase):
    # test to ensure swaps are always improving
    def test_improve(self):
        # generate a knapsack
        cdmkp = CMDKP(n=40, density=70, constraints=1)

        # find a solution
        indices = naive_greedy(cdmkp)

        # check obj val before and after swap
        preSwap = getObjVal(indices, cdmkp.c, cdmkp.C, cdmkp.D)
        indices = perform_swaps(indices, cdmkp)
        postSwap = getObjVal(indices, cdmkp.c, cdmkp.C, cdmkp.D)

        self.assertTrue(postSwap >= preSwap)
