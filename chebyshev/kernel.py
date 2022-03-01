from st.expr import Index, ConstRef
from st.grid import Grid

# Declare indices
i = Index(0)
j = Index(1)
k = Index(2)

# Declare grid
Ac = Grid("Ac", 3)
Ap = Grid("Ap", 3)
Dinv = Grid("Dinv", 3)
RHS = Grid("RHS", 3)
output = Grid("out", 3)

c1 = ConstRef('c1')
c2 = ConstRef('c2')
h2inv = ConstRef('h2inv')

calc = Ac(i, j, k) + c1 * (Ac(i, j, k) + Ap(i, j, k)) + c2 * Dinv(i, j, k) * (RHS(i, j, k) + (Ac(i, j, k) + h2inv * (0.03 * (Ac(i - 1, j - 1, k - 1) + Ac(i + 1, j - 1, k - 1) + Ac(i - 1, j + 1, k - 1) + Ac(i + 1, j + 1, k - 1) + Ac(i - 1, j - 1, k + 1) + Ac(i + 1, j - 1, k + 1) + Ac(i - 1, j + 1, k + 1) + Ac(i + 1, j + 1, k + 1)) + 0.1 * (Ac(i, j - 1, k - 1) + Ac(i - 1, j, k - 1) + Ac(i + 1, j, k - 1) + Ac(i, j + 1, k - 1) + Ac(i - 1, j - 1, k) + Ac(i + 1, j - 1, k) + Ac(i - 1, j + 1, k) + Ac(i + 1, j + 1, k) + Ac(i, j - 1, k + 1) + Ac(i - 1, j, k + 1) + Ac(i + 1, j, k + 1) + Ac(i, j + 1, k + 1)) + 0.46 * (Ac(i, j, k - 1) + Ac(i, j - 1, k) + Ac(i - 1, j, k) + Ac(i + 1, j, k) + Ac(i, j + 1, k) + Ac(i, j, k + 1)) + 4.26 * Ac(i, j, k))))

output(i, j, k).assign(calc)

STENCIL = [output]
