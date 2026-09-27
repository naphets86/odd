"""
Array2D Kernklasse – wird vom Präprozessor verwendet.
Zugriff intern via __getcall__ / __setcall__.
"""

class Array2D:

    def __init__(self, rows: int, cols: int, init=0.0):
        self._rows = rows
        self._cols = cols
        self._data = [float(init)] * (rows * cols)

    def _idx(self, i: int, j: int) -> int:
        if not (0 <= i < self._rows and 0 <= j < self._cols):
            raise IndexError(f"Index ({i},{j}) außerhalb ({self._rows}x{self._cols})")
        return i * self._cols + j

    def __getcall__(self, i: int, j: int) -> float:
        return self._data[self._idx(i, j)]

    def __setcall__(self, i: int, j: int, value):
        self._data[self._idx(i, j)] = float(value)

    # Arithmetik
    def _binop(self, other, op):
        result = Array2D(self._rows, self._cols)
        if isinstance(other, Array2D):
            if self.shape != other.shape:
                raise ValueError("Inkompatible Dimensionen")
            result._data = [op(a, b) for a, b in zip(self._data, other._data)]
        else:
            result._data = [op(a, float(other)) for a in self._data]
        return result

    def __add__ (self, o): return self._binop(o, lambda a,b: a+b)
    def __radd__(self, o): return self._binop(o, lambda a,b: b+a)
    def __sub__ (self, o): return self._binop(o, lambda a,b: a-b)
    def __rsub__(self, o): return self._binop(o, lambda a,b: b-a)
    def __mul__ (self, o): return self._binop(o, lambda a,b: a*b)
    def __rmul__(self, o): return self._binop(o, lambda a,b: b*a)
    def __truediv__ (self, o): return self._binop(o, lambda a,b: a/b)
    def __rtruediv__(self, o): return self._binop(o, lambda a,b: b/a)

    def __iadd__(self, o):
        if isinstance(o, Array2D):
            self._data = [a+b for a,b in zip(self._data, o._data)]
        else:
            self._data = [a+float(o) for a in self._data]
        return self

    def __isub__(self, o):
        if isinstance(o, Array2D):
            self._data = [a-b for a,b in zip(self._data, o._data)]
        else:
            self._data = [a-float(o) for a in self._data]
        return self

    def __imul__(self, o):
        if isinstance(o, Array2D):
            self._data = [a*b for a,b in zip(self._data, o._data)]
        else:
            self._data = [a*float(o) for a in self._data]
        return self

    def __itruediv__(self, o):
        if isinstance(o, Array2D):
            self._data = [a/b for a,b in zip(self._data, o._data)]
        else:
            self._data = [a/float(o) for a in self._data]
        return self

    def __neg__(self):
        result = Array2D(self._rows, self._cols)
        result._data = [-x for x in self._data]
        return result

    def __eq__(self, o):
        if isinstance(o, Array2D):
            return self._data == o._data
        return NotImplemented

    @property
    def shape(self): return (self._rows, self._cols)

    def __repr__(self):
        lines = []
        for i in range(self._rows):
            row = [f"{self.__getcall__(i,j):8.4f}" for j in range(self._cols)]
            lines.append("  [" + "  ".join(row) + "]")
        return f"Array2D({self._rows}x{self._cols}):\n" + "\n".join(lines)


class Array1D:
    """1D-Array mit round-bracket Zugriffssyntax (__getcall__/__setcall__)."""

    def __init__(self, data_or_size, init=0.0):
        if isinstance(data_or_size, (list, tuple)):
            self._data = [float(x) for x in data_or_size]
        else:
            self._data = [float(init)] * int(data_or_size)

    def __getcall__(self, i: int) -> float:
        return self._data[i]

    def __setcall__(self, i: int, value):
        self._data[i] = float(value)

    def __len__(self):
        return len(self._data)

    def __repr__(self):
        return f"Array1D({[round(x, 6) for x in self._data]})"
