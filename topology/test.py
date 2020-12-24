import unittest

import networkx as nx

import topology

class TestLowerStartFiltrationFactory(unittest.TestCase):

    def create_graph(self):
        g = nx.Graph()

        vertex = (
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

        g.add_nodes_from(range(len(vertex)))
        for i, coords in enumerate(vertex):
            g.nodes[i]['pos'] = coords

        g.add_edges_from(edges)

        return g

    def test_create(self):
        direction = (1,0)
        graph = self.create_graph()

        fltr = topology.LowerStarFiltrationFactory(direction).create(graph)
        self.assertTrue(True)

