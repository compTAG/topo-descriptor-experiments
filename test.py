import unittest


import path
import graph

class TestDev(unittest.TestCase):
    def test_playground(self):
        path_manager = path.PathManager('tmp')

        graph.generate_mpeg7_graphs(
            .001,
            path_manager.mpeg7_data_dir,
            path_manager.mpeg7_approx001_graphs,
        )

