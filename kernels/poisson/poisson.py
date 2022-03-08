from st.expr import Index, ConstRef
from st.grid import Grid

# Declare indices
i = Index(0)
j = Index(1)
k = Index(2)

# Declare grid
input = Grid("in", 3)
output = Grid("out", 3)

calc = 2.666 * input(i, j, k) - (0.166 * input(i, j, k - 1) + 0.166 * input(i, j, k + 1) + 0.166 * input(i, j - 1, k) + 0.166 * input(i, j + 1, k) + 0.166 * input(i + 1, j, k) + 0.166 * input(i - 1, j, k)) - (0.0833 * input(i, j - 1, k - 1) + 0.0833 * input(i, j - 1, k + 1) + 0.0833 * input(i, j + 1, k - 1) + 0.0833 * input(i, j + 1, k + 1) + 0.0833 * input(i - 1, j, k - 1) + 0.0833 * input(i - 1, j, k + 1) + 0.0833 * input(i - 1, j - 1, k) + 0.0833 * input(i - 1, j + 1, k) + 0.0833 * input(i + 1, j, k - 1) + 0.0833 * input(i + 1, j, k + 1) + 0.0833 * input(i + 1, j - 1, k) + 0.0833 * input(i + 1, j + 1, k))

output(i, j, k).assign(calc)

STENCIL = [output]