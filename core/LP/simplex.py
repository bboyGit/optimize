import numpy as np
from numpy import linalg

def simplex(coef, A, b):
    """
    Desc: Use simplex method to solve linear programming problem
      The form of linear programming problem is like below:
        Min: coef @ x
        St: Ax >= b and x >= 0
    Parameters:
      coef: A row vector containing coefficients of x.
      A: It's matrix A of Ax >= b.
      b: It's columns vector b of Ax >= b
    Return: A dict like below.
      {'coef': _coef, 'solution': _x, 'cost': _cost}
    """
    # (1) Convert Ax >= b into [A -I]x = b
    nrow = A.shape[0]
    _A = np.concatenate([A, -np.identity(nrow)], axis=1)
    _coef = np.concatenate([coef, np.zeros([1, nrow])], axis=1)

    # (2) Continuously find corner until it's already the optimal point.
    B = _A[:, :nrow]
    N = _A[:, nrow:]
    cb = _coef[:, :nrow]
    cn = _coef[:, nrow:]

    def corner(B, N, cb, cn, b):
        """
        Desc: Find corner and check if it's optimal.
             If it is, then return. Else, update B, N, _coef and do recursive procedure.
        Parameters:
          B: A matrix which is the first m columns of _A. m = nrow of A
          N: A matrix which is the last n columns of _A. n = ncol of A
          cb:  A row vector of coefficients
          cn: A row vector of coefficients. Note that cost is [cb, cn] @ x.
          b:  It's columns vector b of [B, N]x = b. x = [x_B, x_N]'
        Return: The function itself or optimal solution and cost.
        """
        # (1) Calculate the current corner and the object function value of this corner
        m, n = N.shape
        inv_B = linalg.inv(B)
        xb = inv_B @ b
        xn = np.zeros([n, 1])
        cost = cb @ inv_B @ b
        r = cn - cb @ inv_B @ N

        # (2) distinguish whether this corner is optimal.
        cond1 = r >= 0
        if cond1.all():
            # This corner is optimal
            result = {'cb': cb, 'cn': cn, 'xb': xb, 'xn': xn, 'cost': cost}
            return result
        else:
            # It's not the optimal
            enter_idx = r.argmin()
            enter_var = N[:, [enter_idx]]
            ratio = (inv_B @ b)/(inv_B @ enter_var)
            cond2 = ratio < 0
            if cond2.any():
                raise Exception('The object function is unbounded')
            else:
                leave_idx = ratio.argmin()
                leave_var = B[:, [leave_idx]]
                B[:, [leave_idx]], N[:, [enter_idx]] = enter_var, leave_var   # update B and N
                cn_enter = cn[0, enter_idx].copy()
                cb_leave = cb[0, leave_idx].copy()
                cb[0, leave_idx], cn[0, enter_idx] = cn_enter, cb_leave  # update cb and cn
                return corner(B, N, cb, cn, b)

    result = corner(B, N, cb, cn, b)

    return result

# 目前还存在bug，需要再理理算法思路与细节。
