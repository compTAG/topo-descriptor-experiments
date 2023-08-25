import dionysus
import numpy as np
from typing import List, Union, Tuple, Iterable
# rewrite Dot product 
class Dot(object):
    def __init__(self, verts, direction):
        self._verts = np.array([v['pos'] for v in verts])
        self._dir = np.array(direction)

    def __call__(self, idx):
        return np.dot(self._verts[idx], self._dir)
# remove MaxHeight
class MaxHeight(object):
    def __init__(self, heights):
        self._heights = heights

    def __call__(self, simplex):
        return max(self._heights[i] for i in simplex)

class LowerStarFiltrationFactory(object):
    def _compute_function_vals(self, verts):

        for v in verts:
            print(v)
        dot = Dot(verts, self._dir)
        return [dot(v) for v in range(len(verts))]

    def _create_simplices(self, graph, f):
        simp0 = [([v], f[v]) for v in graph.nodes]

        simp1 = [(list(e), max(f[v] for v in e)) for e in graph.edges]

        return simp0 + simp1

    def __init__(self, direction):
        self._dir = direction

    def create(self, graph):
        f = self._compute_function_vals(graph.nodes)
        simplices = self._create_simplices(graph, f)

        filtr = dionysus.Filtration()
        filtr_append = filtr.append  # localize method for faster access
        for vertices, time in simplices:
            simplex = dionysus.Simplex(vertices, time)
            filtr_append(simplex)

        filtr.sort()
        return filtr


class DirectionalDiagram(object):
    def __init__(self, graph, direction):
        filtr = LowerStarFiltrationFactory(direction).create(graph)
        m = dionysus.homology_persistence(filtr)
        self._dgms = dionysus.init_diagrams(m, filtr)
        self.equal_diagrams = []
        self.equal_graphs = []
        self.dir = direction

    def _dgm_equal(self, dgm1, dgm2):
        return len(dgm1) == len(dgm2) and all(x == y for x, y in zip(dgm1, dgm2))

    def __eq__(self, other):
        return self._dgm_equal(self._dgms[0], other._dgms[0]) and \
            self._dgm_equal(self._dgms[1], other._dgms[1])

    def __hash__(self):
        return hash((self._dgms[0], self._dgms[1]))

    def __iter__(self):
        return iter(self._dgms)
'''
class Dot(object):
    def __init__(self, verts, direction):
        self._verts = verts
        self._dir = direction

    def __call__(self, idx):
        v = self._verts[idx]['pos']
        return self._dir[0]*v[0] + self._dir[1]*v[1]


class MaxHeight(object):
    def __init__(self, heights):
        self._heights = heights

    def __call__(self, simplex):
        return max(map(lambda i: self._heights[i], simplex))


class LowerStarFiltrationFactory(object):
    def _compute_function_vals(self, verts):
        for v in verts:
            print(f"Type is {type(v)} and v is {v}\n")
        dot = Dot(verts, self._dir)
        return list(map(dot, verts))

    def _create_simplices(self, graph, f):
        simp0 = map(lambda v: ([v], f[v]), graph.nodes)

        fe = MaxHeight(f)
        simp1 = map(lambda e: (list(e), fe(e)), graph.edges)

        return list(simp0) + list(simp1)

    def __init__(self, direction):
        self._dir = direction

    def create(self, graph):
        f = self._compute_function_vals(graph.nodes)
        simplices = self._create_simplices(graph, f)


        filtr = dionysus.Filtration()
        for vertices, time in simplices:
            simplex = dionysus.Simplex(vertices, time)
            filtr.append(simplex)

        filtr.sort()
        return filtr


class DirectionalDiagram(object):
    def __init__(self, graph, direction):
        filtr = LowerStarFiltrationFactory(direction).create(graph)
        m = dionysus.homology_persistence(filtr)
        self._dgms = dionysus.init_diagrams(m, filtr)
        self.equal_diagrams = []
        self.equal_graphs = []
        self.dir = direction


    def _dgm_equal(self, dgm1, dgm2):
        if len(dgm1) != len(dgm2):
            return False

        for i in range(len(dgm1)):
            if dgm1[i] != dgm2[i]:
                return False

        return True

    def __eq__(self, other):
        return self._dgm_equal(self._dgms[0], other._dgms[0]) and \
            self._dgm_equal(self._dgms[1], other._dgms[1])

    def __hash__(self):
        return hash((self._dgms[0],self._dgms[1]))

    def __iter__(self):
      return (d for d in self._dgms)
'''
