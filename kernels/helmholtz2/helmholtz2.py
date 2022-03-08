from st.expr import Index, ConstRef
from st.grid import Grid

# Declare indices
i = Index(0)
j = Index(1)
k = Index(2)

# Declare grid
x = Grid("x", 3)
alpha = Grid("alpha", 3)
beta_i = Grid("beta_i", 3)
beta_j = Grid("beta_j", 3)
beta_k = Grid("beta_k", 3)
output = Grid("out", 3)

c1 = ConstRef('c[0]')
c2 = ConstRef('c[1]')
h2inv = ConstRef('c[2]')

calc = c1 * alpha(i, j, k) * x(i, j, k) - c2 * h2inv * (beta_i(i + 1, j, k) * (x(i + 1, j, k) - x(i, j, k)) + beta_i(i, j, k) * (x(i - 1, j, k) - x(i, j, k)) + beta_j(i, j + 1, k) * (x(i, j + 1, k) - x(i, j, k)) + beta_j(i, j, k) * (x(i, j - 1, k) - x(i, j, k)) + beta_k(i, j, k + 1) * (x(i, j, k + 1) - x(i, j, k)) + beta_k(i, j, k) * (x(i, j, k - 1) - x(i, j, k)))

output(i, j, k).assign(calc)

STENCIL = [output]