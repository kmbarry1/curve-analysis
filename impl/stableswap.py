from scipy.optimize import fsolve, newton, broyden1
from scipy.stats import gmean

class StableSwap:

    def __init__(self, tokens, A):
        self.tokens = tokens
        N = self.N = len(tokens)
        self.balances = [0] * N
        self.A = A
        self.D = 0.0
        self.supply = 0

    def add_liquidity(self, amounts):
        N = self.N
        assert(len(amounts) == N)

        for i in range(N):
            self.balances[i] += amounts[i]

        old_D = self.D
        new_D = self.calc_D(self.A, self.balances)

        d_token = 0
        if old_D > 0:
            d_token = self.supply * (new_D / old_D - 1)
        else:
            d_token = new_D

        self.D = new_D
        self.supply += d_token

        # return # of LP shares minted
        return d_token

    def exchange(self, i, j, dx):
        N = self.N
        assert(i != j)
        assert(i < N)
        assert(j < N)
        assert(dx > 0)

        dy = self.calc_exchange(i, j, dx)
        assert(dy > 0)
        self.balances[i] += dx
        self.balances[j] -= dy
        return dy

    # side-effect free function for analysis
    def calc_exchange(self, i, j, dx):
        N = self.N
        xp = self.balances.copy()
        xp[i] += dx
        yp = self.calc_y(self.A, xp, self.D, j)
        return xp[j] - yp

    def calc_D(self, A, xp):
        N    = len(xp)
        NN   = N**N
        summ = sum(xp)
        prod = 1.
        for i in range(N):
            prod *= xp[i]
        def f(D):
            return D**(N+1)/(NN*prod) + (A*NN-1)*D - A*NN*summ
        def fp(D):
            return (N+1)*(D**N)/(NN*prod) + (A*NN-1)
        D0 = N * gmean(xp)
        return newton(f, D0, fprime=fp)

    # calculates the new value of xp[j] taking A, D, and all other xp values as given
    def calc_y(self, A, xp, D, j):
        N = len(xp)
        assert(j < N)
        NN = N**N
        summ_ = 0.
        prod_ = 1.
        for i in range(N):
            if i == j:
                continue
            summ_ += xp[i]
            prod_ *= xp[i]
        B = summ_ - D + D/(A*NN)
        C = D**(N+1)/(A*(NN**2)*prod_)
        def g(y):
            return y**2 + B*y + C
        def gp(y):
            return 2*y + B
        y0 = D**N / (prod_ * NN)
        return newton(g, y0, fprime=gp)

    # j is the index of the reference asset
    # r is an array of inverse prices in the reference asset (e.g. if asset i is worth 2 of asset j, r[i] is 0.5)
    # r[j] will be ignored
    #
    # seems to have a bug, and be numerically unstable besides, don't use it
    def calc_bals_given_prices(self, A, D, j, r):
        N = len(r)
        assert(j < N)
        funcs = []
        NN = N**N
        DN1 = D**(N+1)
        def invariant(x):
            summ = 0.
            prod = 1.
            for i in range(N):
                summ += x[i]
                prod *= x[i]
            return DN1/(NN*prod) + (A*NN-1)*D - A*NN*summ
        funcs.append(invariant)
        def deriv_relation(x, i):
            prod2 = 1.
            prod_no_j = 1.
            prod_no_i = 1.
            for k in range(N):
                prod2 *= x[k]
                if k != j:
                    prod_no_j *= x[k]
                if k != i:
                    prod_no_i *= x[k]
            prod2 *= prod2
            return A*NN*(1-r[i]) + DN1*(prod_no_j - r[i]*prod_no_i)/(NN*prod2)
        for i in range(N):
            if i == j:
                continue
            funcs.append(lambda x: deriv_relation(x, i))
        def F(x):
            result = []
            for i in range(len(x)):
                result.append(funcs[i](x))
            return result
        return broyden1(F, [D/N]*N)
    
    def get_virtual_price(self):
        D = self.calc_D(self.A, self.balances)
        return D / self.supply
