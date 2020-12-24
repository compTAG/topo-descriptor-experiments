import unittest

import networkx as nx

import topology

def create_graph(verts, edges):
    g = nx.Graph()
    g.add_nodes_from(range(len(verts)))
    for i, coords in enumerate(verts):
        g.nodes[i]['pos'] = coords

    g.add_edges_from(edges)

    return g


class TestLowerStartFiltrationFactory(unittest.TestCase):
    def assertEqualEdge(self, e, s):
        self.assertTrue(len(s) == 2 and e == (s[0], s[1]))

    def assertEqualVert(self, v, s):
        self.assertTrue(len(s) == 1 and v == s[0])

    def test_create(self):
        verts = (
            (0,0),
            (2,2),
            (1,3),
            (4,1),
            (5,2),
            (6,1),
        )

        edges = (
            (0, 1),
            (1, 2),
            (2, 0),
            (1, 3),
            (3, 5),
            (5, 4),
            (4, 3),
        )
        graph = create_graph(verts, edges)
        direction = (1,0)
        fltr = topology.LowerStarFiltrationFactory(direction).create(graph)

        self.assertEqualVert(0, fltr[0])
        self.assertEqualVert(2, fltr[1])
        self.assertEqualEdge((0, 2), fltr[2])
        self.assertEqualVert(1, fltr[3])
        self.assertEqualEdge((0, 1), fltr[4])
        self.assertEqualEdge((1, 2), fltr[5])
        self.assertEqualVert(3, fltr[6])
        self.assertEqualEdge((1, 3), fltr[7])
        self.assertEqualVert(4, fltr[8])
        self.assertEqualEdge((3, 4), fltr[9])
        self.assertEqualVert(5, fltr[10])
        self.assertEqualEdge((3, 5), fltr[11])
        self.assertEqualEdge((4, 5), fltr[12])


class TestDirectionDgm(unittest.TestCase):

    def test_setup(self):
        self.assertTrue(True)
