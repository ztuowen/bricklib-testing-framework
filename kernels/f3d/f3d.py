from st.expr import Index, ConstRef
from st.grid import Grid

SIZE = $SIZE

# Declare indices
i = Index(0)
j = Index(1)
k = Index(2)

coeffs = []
for a in range(0, 8):
    coeffs.append([])
    for b in range(0, 8):
        coeffs[a].append([])
        for c in range(0, 8):
            coeffs[a][b].append(ConstRef(f"c[{a}][{b}][{c}]"))

# Declare grid
input = Grid("bIn", 3)
output = Grid("bOut", 3)

base = 0
for i_d in range(-RADIUS, RADIUS + 1):
    for j_d in range(-RADIUS, RADIUS + 1):
        for k_d in range(-RADIUS, RADIUS + 1):
            base += input(i_d + i, j_d + j, k_d + k) * coeffs[i_d + RADIUS][j_d + RADIUS][k_d + RADIUS]
output(i, j, k).assign(base)
STENCIL = [output]
