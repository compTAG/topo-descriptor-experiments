import dionysus
import matplotlib.pyplot as plt
import numpy as np


class Dot(object):
    """
    Class representing a dot product computation based on a given direction.

    This class computes the dot product of a vertex position with a given direction vector.

    Attributes:
        _verts (list): List of vertices.
        _dir (tuple): Tuple representing the direction vector.

    Methods:
        __init__: Initialize the Dot object.
        __call__: Compute the dot product for a given vertex index.
    """
    def __init__(self, verts, direction):
        """
        Initialize the Dot object.

        Parameters:
            verts (list): List of vertices with positions.
            direction (tuple): Tuple representing the direction vector.

        Returns:
            None
        """
        self._verts = verts
        self._dir = direction

    def __call__(self, idx):
        """
        Compute the dot product for a given vertex index.

        Parameters:
            idx (int): Index of the vertex.

        Returns:
            float: Dot product value.
        """
        v = self._verts[idx]['pos']
        return self._dir[0]*v[0] + self._dir[1]*v[1]


class MaxHeight(object):
    """
    Class representing a function to compute the maximum height of a simplex.

    This class computes the maximum height of a simplex based on a given list of heights.

    Attributes:
        _heights (list): List of heights associated with vertices.

    Methods:
        __init__: Initialize the MaxHeight object.
        __call__: Compute the maximum height for a given simplex.
    """
    def __init__(self, heights):
        """
        Initialize the MaxHeight object.

        Parameters:
            heights (list): List of heights associated with vertices.

        Returns:
            None
        """
        self._heights = heights

    def __call__(self, simplex):
        """
        Compute the maximum height for a given simplex.

        Parameters:
            simplex (list): List of vertex indices forming the simplex.

        Returns:
            float: Maximum height of the simplex.
        """
        max_height = float('-inf') # Initialize maximum height
        for i in simplex:
            if self._heights[i] > max_height:
                max_height = self._heights[i] # Update maximum height if needed
        return max_height


class LowerStarFiltrationFactory(object):
    """
    Class for creating a filtration based on lower star complexes.

    This class generates a filtration from a graph using lower star complexes based on a given direction.

    Attributes:
        _dir (tuple): Tuple representing the direction vector.

    Methods:
        _compute_function_vals: Compute function values for vertices based on dot product with direction.
        _create_simplices: Create simplices for the filtration.
        __init__: Initialize the LowerStarFiltrationFactory object.
        create: Create a filtration for the given graph.
    """
    def _compute_function_vals(self, verts):
        """
        Compute function values for vertices based on dot product with direction.

        Parameters:
            verts (list): List of vertices with positions.

        Returns:
            list: List of function values for vertices.
        """
        dot = Dot(verts, self._dir)
        result = []
        for vert in verts:
            result.append(dot(vert))
        return result

    def _create_simplices(self, graph, f):
        """
        Create simplices for the filtration.

        Parameters:
            graph: The input graph.
            f (list): List of function values for vertices.

        Returns:
            list: List of simplices for the filtration.
        """
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
        """
        Initialize the LowerStarFiltrationFactory object.

        Parameters:
            direction (tuple): Tuple representing the direction vector.

        Returns:
            None
        """
        self._dir = direction

    def create(self, graph):
        """
        Create a filtration for the given graph.

        Parameters:
            graph: The input graph.

        Returns:
            dionysus.Filtration: Filtration generated using lower star complexes.
        """
        f = self._compute_function_vals(graph.nodes)
        simplices = self._create_simplices(graph, f)


        filtr = dionysus.Filtration()
        for vertices, time in simplices:
            filtr.append(dionysus.Simplex(vertices, time))

        return filtr


class DirectionalDiagram(object):
    """
    Class representing a directional diagram generated from a graph.

    This class is used to create and manipulate directional diagrams based on a given direction.

    Attributes:
        _dgms (list): List of diagrams generated from the homology of the filtration.
        equal_diagrams (list): List of equal diagrams.
        equal_graphs (list): List of graphs with equal diagrams.
        dir (tuple): Tuple representing the direction of the diagram.

    Methods:
        __init__: Initialize the DirectionalDiagram object.
        _dgm_equal: Check if two diagrams are equal.
        __eq__: Check equality between two DirectionalDiagram objects.
        __hash__: Generate a hash value for the DirectionalDiagram object.
        __iter__: Iterate over the diagrams in the DirectionalDiagram.
    """

    def __init__(self, graph, direction):
        """
        Initialize the DirectionalDiagram object.

        Parameters:
            graph: The input graph.
            direction (tuple): Tuple representing the direction.

        Returns:
            None
        """
        filtr = LowerStarFiltrationFactory(direction).create(graph)
        m = dionysus.homology_persistence(filtr)
        self._dgms = dionysus.init_diagrams(m, filtr)
        self.equal_diagrams = []
        self.equal_graphs = []
        self.dir = direction


    def _dgm_equal(self, dgm1, dgm2): 
        """
        Check if two diagrams are equal.

        Parameters:
            dgm1: First diagram.
            dgm2: Second diagram.

        Returns:
            bool: True if the diagrams are equal, False otherwise.
        """

        if dionysus.bottleneck_distance(dgm1,dgm2) != 0:
            return False

        return True

    def __eq__(self, other):
        """
        Check equality between two DirectionalDiagram objects.

        Parameters:
            other: The other DirectionalDiagram object.

        Returns:
            bool: True if the diagrams are equal, False otherwise.
        """
        return self._dgm_equal(self._dgms[0], other._dgms[0]) and \
            self._dgm_equal(self._dgms[1], other._dgms[1])

    def __hash__(self):
        """
        Generate a hash value for the DirectionalDiagram object.

        Returns:
            int: Hash value.
        """
        return hash((self._dgms[0],self._dgms[1]))

    def __iter__(self):
      """
      Iterate over the diagrams in the DirectionalDiagram.

        Returns:
            iterator: Iterator over diagrams.
      """
      return (d for d in self._dgms)
