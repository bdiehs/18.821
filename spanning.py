import numpy as np

EPSILON = 1e-9
def equalish(x, y):
    return all(abs(x - y) < EPSILON)

def first_nonzero(x):
    return next((i for i, k in enumerate(x) if k != 0), len(x))

class SpanningSet:
    def __init__(self, initial_bases):
        self.basis = [x.flatten() for x in initial_bases]
        self.set = initial_bases
        self.pivot_indices = []

        self._row_echelon()


    def __iter__(self):
        yield from self.set

    '''
    Put our own basis in row-echelon form.
    '''
    def _row_echelon(self):
        old_basis = self.basis
        ut_basis = []

        while len(old_basis) > 0:
            old_basis = sorted(old_basis, key = first_nonzero)

            pivot_index = first_nonzero(old_basis[0])

            pivot = old_basis[0] / old_basis[0][pivot_index]

            ut_basis.append(pivot)

            old_basis = [
                row - pivot * row[pivot_index]
                for row in old_basis[1:]
            ]

        ut_basis = list(reversed(ut_basis))

        echelon_basis = []

        while len(ut_basis) > 0:
            pivot = ut_basis[0]
            pivot_index = first_nonzero(pivot)

            echelon_basis.append(pivot)

            ut_basis = [x - pivot * x[pivot_index] for x in ut_basis[1:]]

        self.basis = echelon_basis
        self.pivot_indices = [first_nonzero(x) for x in self.basis]


    def spans(self, candidate):
        candidate = candidate.flatten()

        coefficients = [candidate[index] for index in self.pivot_indices]

        if equalish(candidate, sum(coefficients[i] * self.basis[i] for i in range(len(self.basis)))):
            return True
        else:
            return False

    def add(self, candidate):
        self.basis.append(candidate.flatten())
        self.set.append(candidate)
        self._row_echelon()

    def full_rank(self):
        return len(self.basis) == len(self.basis[0])

'''
Some tests of SpanningSet
'''
'''
DIM = 2
x = SpanningSet([np.eye(DIM)])
print(x.basis)

y, z, w = [np.random.rand(DIM, DIM) for _ in range(3)]
print(y)
x.add(y)
print(x.basis)
print(z)
x.add(z)
print(x.basis)
print(w)
x.add(w)
print(x.basis)
'''

def test_generation(X, Y):
    current_spanning_set = SpanningSet([np.eye(X.shape[0])])

    mult_strings = ['']
    candidates = []

    degree = 0
    while True:
        new_candidates = [X @ z for z in current_spanning_set] + [Y @ z for z in current_spanning_set]
        new_candidate_strings = ['X' + z for z in mult_strings] + ['Y' + z for z in mult_strings]

        any_new = False
        for i, candidate in enumerate(new_candidates):
            if not current_spanning_set.spans(candidate):

                current_spanning_set.add(candidate)
                mult_strings.append(new_candidate_strings[i])
                candidates.append(candidate)

                any_new = True

        if not any_new:
            break
        
        degree += 1

    if current_spanning_set.full_rank():
        return degree, mult_strings, candidates
    else:
        return False

def cyclic_shift_dimension(n):
    initial_array = [[0 for _ in range(n)] for __ in range(n)]
    for i in range(n):
        initial_array[i][(i + 1) % n] = 1
    return np.array(initial_array)

def diverse_diagonal_dimension(n):
    initial_array = [[0 for _ in range(n)] for __ in range(n)]
    for i in range(n):
        initial_array[i][i] = i
    return np.array(initial_array)

for n in range(1, 20):
    d, m, c = test_generation(cyclic_shift_dimension(n), diverse_diagonal_dimension(n))

    print('DIMENSION %d, DEGREE %d' % (n, d))
