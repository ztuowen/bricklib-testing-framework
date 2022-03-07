from st.expr import Index, ConstRef
from st.grid import Grid

SIZE = $SIZE

# Declare indices
i = Index(0)
j = Index(1)

param = [
       ConstRef("dev_coeff[0]"),
       ConstRef("dev_coeff[1]"),
       ConstRef("dev_coeff[2]"),
       ConstRef("dev_coeff[3]"),
       ConstRef("dev_coeff[4]"),
       ConstRef("dev_coeff[5]"),
       ConstRef("dev_coeff[6]"),
       ConstRef("dev_coeff[7]"),
       ConstRef("dev_coeff[8]")
]

# Declare grid
input = Grid("bIn", 2)
output = Grid("bOut", 2)

calc = param[0] * input(i, j)
for a in range(1, SIZE + 1):
    calc = calc + param[a] * ( \
            input(i, j + a) + input(i + a, j) + \
            input(i, j - a) + input(i - a, j))
output(i, j).assign(calc)
STENCIL = [output]
