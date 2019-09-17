import numpy as np
import scipy.linalg

EPSILON = 1e-9
def equalish(x, y):
    return all(abs(x - y) < EPSILON)

def first_nonzero(x):
    return next((i for i, k in enumerate(x) if k != 0), len(x))

def singleton(n, i, j):
    initial_array = [[0 for _ in range(n)] for __ in range(n)]
    initial_array[i][j] = 1
    return np.array(initial_array)

class SpanningSet:
    def __init__(self, initial_bases, initial_strings):
        self.basis = [x.flatten() for x in initial_bases]
        self.set = initial_bases
        self.strings = initial_strings
        self.pivot_indices = []

        self._row_echelon()


    def __iter__(self):
        yield from self.set

    '''
    Put our own basis in row-echelon form.
    '''
    def _row_echelon(self):
        '''
        old_basis = self.basis
        ut_basis = []

        while len(old_basis) > 0:
            old_basis = list(sorted(old_basis, key = first_nonzero))

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
        '''

        _, __, self.basis = scipy.linalg.lu(np.array(self.basis))

        pivots = [first_nonzero(x) for x in self.basis]
        self.pivot_indices = pivots

        for i in range(len(self.basis)):
            self.basis[i] /= self.basis[i][pivots[i]]

        for i in reversed(range(len(self.basis))):
            for j in reversed(range(i)):
                self.basis[j] -= self.basis[i] * self.basis[j][pivots[i]]

        self.basis = list(self.basis)

    def terms(self, expr):
        if not self.full_rank():
            raise Exception("I'm singular!")

        M = np.transpose(np.array([x.flatten() for x in self.set]))
        print(M)
        print(np.linalg.det(M))
        #print(M)
        Minv = np.linalg.inv(M)

        coeffs = Minv @ expr.flatten()

        return ' + '.join(['(%f %s)' % (coeffs[i], self.strings[i]) for i in range(len(coeffs)) if coeffs[i] != 0])
        #return ' + '.join(['(%f %r)' % (coeffs[i], self.set[i]) for i in range(len(coeffs)) if coeffs[i] != 0])

    def spans(self, candidate):
        candidate = candidate.flatten()

        coefficients = [candidate[index] for index in self.pivot_indices]

        if equalish(candidate, sum(coefficients[i] * self.basis[i] for i in range(len(self.basis)))):
            return True
        else:
            return False

    def add(self, candidate, string):
        self.basis.append(candidate.flatten())
        self.set.append(candidate)
        self.strings.append(string)
        self._row_echelon()

    def full_rank(self):
        return len(self.basis) == len(self.basis[0])

'''
Some tests of SpanningSet
'''
DIM = 5
x = SpanningSet([np.eye(DIM)], ['I'])
print(x.basis)

l = [np.random.rand(DIM, DIM) for _ in range(100)]

for i, y in enumerate(l):
    print(y)
    print(x.spans(y))
    if not x.spans(y):
        x.add(y, 'y%d' % i)
        print(x.basis)
    print('------')

for z in l:
    print(x.spans(z))
    print(x.terms(z))

def test_generation(X, Y):
    current_spanning_set = SpanningSet([np.eye(X.shape[0])], ['I'])

    mult_strings = ['']
    candidates = []

    degree = 0
    while True:
        new_candidates = [X @ z for z in current_spanning_set] + [Y @ z for z in current_spanning_set]
        new_candidate_strings = ['X' + z for z in mult_strings] + ['Y' + z for z in mult_strings]

        print(len(new_candidate_strings))

        any_new = False
        for i, candidate in enumerate(new_candidates):
            if not current_spanning_set.spans(candidate):
                print('ADDING')
                print(new_candidate_strings[i])
                print(len(mult_strings))
                print(len(set(mult_strings)))

                current_spanning_set.add(candidate, new_candidate_strings[i])
                mult_strings.append(new_candidate_strings[i])
                candidates.append(candidate)

                any_new = True

        if not any_new:
            break
        
        degree += 1

    if current_spanning_set.full_rank():
        return degree, current_spanning_set
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
        initial_array[i][i] = i + 1
    return np.array(initial_array)

'''
for n in range(1, ):
    X = np.random.rand
    d, S = test_generation(cyclic_shift_dimension(n), diverse_diagonal_dimension(n))

    print('DIMENSION %d, DEGREE %d' % (n, d))
    print('BECAUSE:')
    for i in range(n):
        for j in range(n):
            X = singleton(n, i, j) * 6
            print('%r = %s' % (X, S.terms(X)))
    #for mi, ci in zip(m, c):
    #    print('%s = %r' % (mi, ci))
'''

ITERS = 20
n = 10
for _ in range(ITERS):
    X, Y = diverse_diagonal_dimension(n), np.round(10 * np.random.rand(n, n))

    print('X = %r' % X)
    print('Y = %r' % Y)

    d, S = test_generation(X, Y)

    print('DEGREE %d' % d)
    print('BECAUSE:')
    for i in range(n):
        for j in range(n):
            X = singleton(n, i, j)
            print('%r = %s' % (X, S.terms(X)))
