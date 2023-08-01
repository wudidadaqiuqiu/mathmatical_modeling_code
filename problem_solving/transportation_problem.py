import cvxpy as cp
import numpy as np
from numpy import ndarray


def balance_transportation_problem(cost_mat: ndarray, production: ndarray, sales_volume: ndarray) -> tuple:
    assert np.sum(production) == np.sum(sales_volume)
    x = cp.Variable(cost_mat.shape, pos=True)
    obj = cp.Minimize(cp.sum(cp.multiply(cost_mat, x)))
    cons = [x >= 0,
            cp.sum(x, axis=1) == production,
            cp.sum(x, axis=0) == sales_volume]
    prob = cp.Problem(obj, cons)
    prob.solve(solver='GLPK_MI')
    return x.value, prob.value