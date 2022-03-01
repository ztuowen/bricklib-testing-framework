out(i, j, k) =
    Ac(i, j, k) + c1 * (Ac(i, j, k) + Ap(i, j, k)) +
    c2 * Dinv(i, j, k) *
        (RHS(i, j, k) +
         (Ac(i, j, k) +
          h2inv *
              (0.03 * (Ac(i - 1, j - 1, k - 1) + Ac(i + 1, j - 1, k - 1) +
                       Ac(i - 1, j + 1, k - 1) + Ac(i + 1, j + 1, k - 1) +
                       Ac(i - 1, j - 1, k + 1) + Ac(i + 1, j - 1, k + 1) +
                       Ac(i - 1, j + 1, k + 1) + Ac(i + 1, j + 1, k + 1)) +
               0.1 * (Ac(i, j - 1, k - 1) + Ac(i - 1, j, k - 1) +
                      Ac(i + 1, j, k - 1) + Ac(i, j + 1, k - 1) +
                      Ac(i - 1, j - 1, k) + Ac(i + 1, j - 1, k) +
                      Ac(i - 1, j + 1, k) + Ac(i + 1, j + 1, k) +
                      Ac(i, j - 1, k + 1) + Ac(i - 1, j, k + 1) +
                      Ac(i + 1, j, k + 1) + Ac(i, j + 1, k + 1)) +
               0.46 * (Ac(i, j, k - 1) + Ac(i, j - 1, k) + Ac(i - 1, j, k) +
                       Ac(i + 1, j, k) + Ac(i, j + 1, k) + Ac(i, j, k + 1)) +
               4.26 * Ac(i, j, k))));