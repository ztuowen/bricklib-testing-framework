from st.expr import Index, ConstRef
from st.grid import Grid

RADIUS = $SIZE

# indices
i = Index(0)
j = Index(1)
k = Index(2)

coeffs = []
for a in range(0, 4):
    coeffs.append([])
    for b in range(0, 4):
        coeffs[a].append([])
        for c in range(0, 4):
            coeffs[a][b].append(ConstRef(f"coeff[{a}][{b}][{c}]"))

Cs = [
    ConstRef("c[0]"),
    ConstRef("c[1]"),
    ConstRef("c[2]")
]

ac = Grid("Ac", 3)
ap = Grid("Ap", 3)
dinv = Grid("Dinv", 3)
rhs = Grid("RHS", 3)
out = Grid("out", 3)

base = 0
for i_d in range(-RADIUS, RADIUS + 1):
    for j_d in range(-RADIUS, RADIUS + 1):
        for k_d in range(-RADIUS, RADIUS + 1):
            base += coeffs(i_d + RADIUS, j_d + RADIUS, k_d + RADIUS) * ac(i + i_d, j + j_d, k + k_d)
final = ac(i, j, k) + cs[0] * (ac(i, j, k) + ap(i, j, k)) + c[1] * dinv(i, j, k) * (rhs(i, j, k) + (ac(i, j, k) + c[2] * base))
out(i, j, k).assign(final)
STENCIL = [out]
