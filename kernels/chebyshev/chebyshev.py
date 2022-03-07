from st.expr import Index, ConstRef
from st.grid import Grid

# indices
i = Index(0)
j = Index(1)
k = Index(2)

Ac = Grid("Ac", 3)
Ap = Grid("Ap", 3)
Dinv = Grid("Dinv", 3)
RHS = Grid("RHS", 3)
out = Grid("out", 3)

c1 = ConstRef('c[0]')
c2 = ConstRef('c[1]')
h2inv = ConstRef('c[2]')

calc = Ac(i, j, k) + c1 * (Ac(i, j, k) + Ap(i, j, k)) + \
    c2 * Dinv(i, j, k) * \
    (RHS(i, j, k) + \
    (Ac(i, j, k) + \
    h2inv * ( \
        0.03 * (Ac(i - 1, j - 1, k - 1) + Ac(i + 1, j - 1, k - 1) + \
                Ac(i - 1, j + 1, k - 1) + Ac(i + 1, j + 1, k - 1) + \
                Ac(i - 1, j - 1, k + 1) + Ac(i + 1, j - 1, k + 1) + \
                Ac(i - 1, j + 1, k + 1) + Ac(i + 1, j + 1, k + 1)) + \
        0.1 * (Ac(i, j - 1, k - 1) + Ac(i - 1, j, k - 1) + \
                Ac(i + 1, j, k - 1) + Ac(i, j + 1, k - 1) + \
                Ac(i - 1, j - 1, k) + Ac(i + 1, j - 1, k) + \
                Ac(i - 1, j + 1, k) + Ac(i + 1, j + 1, k) + \
                Ac(i, j - 1, k + 1) + Ac(i - 1, j, k + 1) + \
                Ac(i + 1, j, k + 1) + Ac(i, j + 1, k + 1)) + \
        0.46 * (Ac(i, j, k - 1) + Ac(i, j - 1, k) + Ac(i - 1, j, k) + \
                Ac(i, j, k + 1) + Ac(i, j + 1, k) + Ac(i + 1, j, k)) + \
        4.26 * Ac(i, j, k) \
        )))
STENCIL = [out]
