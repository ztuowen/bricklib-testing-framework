from st.expr import Index, ConstRef
from st.grid import Grid

SIZE = $SIZE

# Declare indices
i = Index(0)
j = Index(1)

coeffs = []
for a in range(0, 8):
    coeffs.append([])
    for b in range(0, 8):
        coeffs[a][b].append(ConstRef(f"c[{a}][{b}]"))

# Declare grid
input = Grid("bIn", 3)
output = Grid("bOut", 3)

base = 0
for i_d in range(-SIZE, SIZE + 1):
    for j_d in range(-SIZE, SIZE + 1):
        base += input(i_d + i, j_d + j) * coeffs[i_d + SIZE][j_d + SIZE]
output(i, j).assign(base)
STENCIL = [output]
