import dionysus


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

