import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from array2d_core import Array1D, Array2D
# Crout LU-Zerlegung – geschrieben in py2-Syntax
# Zuweisung:  a(i,j) = expr   (echte = Zuweisung mit runden Klammern)
# Verbundzuweisung:  b(j) -= expr  wird ebenfalls unterstützt

def lu_decomposition(a, n):
    a.__setcall__(2,0, a.__getcall__(2,0) / a.__getcall__(1,1))

    for j in range(2, n):
        a.__setcall__(j,  1, a.__getcall__(j,1) - a.__getcall__(j,0) * a.__getcall__(j-1,2))
        a.__setcall__(j+1,0, a.__getcall__(j+1,0) / a.__getcall__(j,1))

    a.__setcall__(n,1, a.__getcall__(n,1) - a.__getcall__(n,0) * a.__getcall__(n-1,2))


def solve_system(a, b, n):
    for j in range(2, n + 1):
        b.__setcall__(j, b.__getcall__(j) - (a.__getcall__(j,0) * b.__getcall__(j-1)))

    b.__setcall__(n, b.__getcall__(n) / (a.__getcall__(n,1)))

    for j in range(n - 1, 0, -1):
        b.__setcall__(j, b.__getcall__(j) - (a.__getcall__(j,2) * b.__getcall__(j+1)))
        b.__setcall__(j, b.__getcall__(j) / (a.__getcall__(j,1)))


# Test
a = Array2D(5, 3)
a.__setcall__(1,1, 2.0)
a.__setcall__(1,2, -1.0)
a.__setcall__(2,0, -1.0)
a.__setcall__(2,1, 2.0)
a.__setcall__(2,2, -1.0)
a.__setcall__(3,0, -1.0)
a.__setcall__(3,1, 2.0)

b = Array1D([0.0, 1.0, 0.0, 1.0])

print("Matrix vor LU:")
print(a)

lu_decomposition(a, 3)
solve_system(a, b, 3)

print("\nLoesung:", [round(b.__getcall__(k), 6) for k in range(1, 4)])
print("Erwartet: [1.0, 1.0, 1.0]")
