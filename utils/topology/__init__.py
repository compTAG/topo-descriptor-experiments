import dionysus
import matplotlib.pyplot as plt
import numpy as np


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
        max_height = float('-inf')
        for i in simplex:
            if self._heights[i] > max_height:
                max_height = self._heights[i]
        return max_height


class LowerStarFiltrationFactory(object):
    def _compute_function_vals(self, verts):
        dot = Dot(verts, self._dir)
        result = []
        for vert in verts:
            result.append(dot(vert))
        return result

    def _create_simplices(self, graph, f):
        simp0 = []
        for v in graph.nodes:
            simp0.append(([v], f[v]))

        fe = MaxHeight(f)
        simp1 = []
        for e in graph.edges:
            simp1.append((list(e), fe(e)))
        
        simplices = []
        
        for simp in simp0:
            simplices.append(simp)
        
        for simp in simp1:
            simplices.append(simp)

        return simplices

    def __init__(self, direction):
        self._dir = direction

    def create(self, graph):
        f = self._compute_function_vals(graph.nodes)
        simplices = self._create_simplices(graph, f)


        filtr = dionysus.Filtration()
        for vertices, time in simplices:
            filtr.append(dionysus.Simplex(vertices, time))

        return filtr


class DirectionalDiagram(object):
    def __init__(self, graph, direction):
        filtr = LowerStarFiltrationFactory(direction).create(graph)
        m = dionysus.homology_persistence(filtr)
        self._dgms = dionysus.init_diagrams(m, filtr)
        """
        print(self._dgms)
        for i, dgm in enumerate(self._dgms):
            for pt in dgm:
                print(f"Diagram: {i}; Birth: {pt.birth}; Death: {pt.death}")

        
        CODE TO VISUALIZE PD'S

        birth = [p.birth for p in self._dgms[0]]
        death = [p.death for p in self._dgms[0]]

        print(f"Birth:{birth}\nDeath:{death} for First diagram")


       # Assuming dgms is a list of persistence diagrams
        for i, dgm in enumerate(self._dgms):
            if len(dgm) > 0:
                for p in dgm:
                    if np.isfinite(p.death):
                        plt.plot([p.birth, p.death], [p.birth, p.death], 'b-')  # Plot persistence point
                    else:
                        plt.axvline(x=p.birth, color='b', linestyle='--')  # Plot vertical line starting at birth
            else:
                plt.axvline(x=0, color='r', linestyle='--')  # Plot vertical line for empty diagram

        plt.xlabel('Birth')
        plt.ylabel('Death')
        plt.title('Persistence Diagrams')
        plt.show()


        """

        self.equal_diagrams = []
        self.equal_graphs = []
        self.dir = direction


    def _dgm_equal(self, dgm1, dgm2): #use a direction to check equality 
        
        if dionysus.bottleneck_distance(dgm1,dgm2) != 0:
            return False

        return True

    def __eq__(self, other):
        return self._dgm_equal(self._dgms[0], other._dgms[0]) and \
            self._dgm_equal(self._dgms[1], other._dgms[1])

    def __hash__(self):
        return hash((self._dgms[0],self._dgms[1]))

    def __iter__(self):
      return (d for d in self._dgms)
