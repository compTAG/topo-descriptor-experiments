import os



GRAPHS = 'graphs'
GRAPHS_001_APPROX = 'graphs_001_approx'
GRAPHS_005_APPROX = 'graphs_005_approx'

ANALYSIS_001_APPROX = 'analysis_001_approx'
ANALYSIS_005_APPROX = 'analysis_005_approx'
MPEG7 = 'mpeg7'
MNIST = 'mnist'
RANDOM = 'random'
FIGS = 'figs'


dir_data = 'data'

dir_list = [
    dir_data,
    os.path.join('graphs_005_approx','mnist'),
    os.path.join('graphs_005_approx','mnist_imgs'),
    os.path.join('graphs_001_approx', 'mnist'),
    os.path.join('graphs_001_approx', 'mnist_imgs'),
    os.path.join('graphs_005_approx','mpeg7'),
    os.path.join('graphs_005_approx', 'mpeg7_imgs'),
    os.path.join('graphs_005_approx', 'mpeg7_extra'),
    os.path.join('graphs_001_approx', 'mpeg7'),
    os.path.join('graphs_001_approx', 'mpeg7_imgs'),
    os.path.join('graphs_001_approx', 'mpeg7_extra'),
    os.path.join('graphs', 'random_imgs'),
    os.path.join('graphs', 'random'),
    os.path.join('analysis_005_approx', 'smallest_angle_exp', 'combined_data', 'mnist'),
    os.path.join('analysis_005_approx', 'smallest_angle_exp', 'combined_data', 'mpeg7'),
    os.path.join('analysis_005_approx', 'smallest_angle_exp', 'combined_data', 'random'),
    os.path.join('analysis_001_approx', 'smallest_angle_exp', 'combined_data', 'mnist'),
    os.path.join('analysis_001_approx', 'smallest_angle_exp', 'combined_data', 'mpeg7'),
    os.path.join('analysis_001_approx', 'smallest_angle_exp', 'combined_data', 'random'),
    os.path.join('analysis_005_approx', 'uniform_sample_exp', 'combined_data', 'mnist'),
    os.path.join('analysis_005_approx', 'uniform_sample_exp', 'combined_data', 'mpeg7'),
    os.path.join('analysis_005_approx', 'uniform_sample_exp', 'combined_data', 'random'),
    os.path.join('analysis_001_approx', 'uniform_sample_exp', 'combined_data', 'mnist'),
    os.path.join('analysis_001_approx', 'uniform_sample_exp', 'combined_data', 'mpeg7'),
    os.path.join('analysis_001_approx', 'uniform_sample_exp', 'combined_data', 'random'),
    os.path.join('figs', 'smallest_angle_exp', 'random'),
    os.path.join('figs', 'smallest_angle_exp', 'mnist'),
    os.path.join('figs', 'smallest_angle_exp', 'mpeg7'),
    os.path.join('figs', 'uniform_sample_exp', 'random'),
    os.path.join('figs', 'uniform_sample_exp', 'mnist'),
    os.path.join('figs', 'uniform_sample_exp', 'mpeg7'),
]






    # os.path.join('graphs_005_approx','mnist'),
    # os.path.join('graphs_005_approx','mnist_imgs'),
    # os.path.join('graphs_001_approx', 'mnist'),
    # os.path.join('graphs_001_approx', 'mnist_imgs'),
    # os.path.join('graphs_005_approx','mpeg7'),
    # os.path.join('graphs_005_approx', 'mpeg7_imgs'),
    # os.path.join('graphs_005_approx', 'mpeg7_extra'),
    # os.path.join('graphs_001_approx', 'mpeg7'),
    # os.path.join('graphs_001_approx', 'mpeg7_imgs'),
    # os.path.join('graphs_001_approx', 'mpeg7_extra'),
    # os.path.join('graphs', 'random_imgs'),
    # os.path.join('graphs', 'random'),
    # os.path.join('analysis_005_approx', 'smallest_angle_exp', 'combined_data', 'mnist'),
    # os.path.join('analysis_005_approx', 'smallest_angle_exp', 'combined_data', 'mpeg7'),
    # os.path.join('analysis_005_approx', 'smallest_angle_exp', 'combined_data', 'random'),
    # os.path.join('analysis_001_approx', 'smallest_angle_exp', 'combined_data', 'mnist'),
    # os.path.join('analysis_001_approx', 'smallest_angle_exp', 'combined_data', 'mpeg7'),
    # os.path.join('analysis_001_approx', 'smallest_angle_exp', 'combined_data', 'random'),
    # os.path.join('analysis_005_approx', 'uniform_sample_exp', 'combined_data', 'mnist'),
    # os.path.join('analysis_005_approx', 'uniform_sample_exp', 'combined_data', 'mpeg7'),
    # os.path.join('analysis_005_approx', 'uniform_sample_exp', 'combined_data', 'random'),
    # os.path.join('analysis_001_approx', 'uniform_sample_exp', 'combined_data', 'mnist'),
    # os.path.join('analysis_001_approx', 'uniform_sample_exp', 'combined_data', 'mpeg7'),
    # os.path.join('analysis_001_approx', 'uniform_sample_exp', 'combined_data', 'random'),
    # os.path.join('figs', 'smallest_angle_exp', 'random'),
    # os.path.join('figs', 'smallest_angle_exp', 'mnist'),
    # os.path.join('figs', 'smallest_angle_exp', 'mpeg7'),
    # os.path.join('figs', 'uniform_sample_exp', 'random'),
    # os.path.join('figs', 'uniform_sample_exp', 'mnist'),
    # os.path.join('figs', 'uniform_sample_exp', 'mpeg7'),
    #


class FolderMaker(object):
    def make_folders(self, dir_list):
      for path in dir_list:
        self.make_folder(path)

    def make_folder(self, folder_path):
      if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print("Directory %s created" % folder_path)
      else:
        print("Directory %s already exists" % folder_path)


class GraphPath(object):
    def __init__(self, base):
        self.base = base

    @property
    def graph_dir(self):
        return os.path.join(self.base, 'graph')

    @property
    def image_dir(self):
        return os.path.join(self.base, 'image')

    @property
    def not_used_filename(self):
        return os.path.join(self.base, 'not-used.txt')


    def make_dirs(self):
        FolderMaker().make_folders([
            self.base,
            self.graph_dir,
            self.image_dir,
        ])


class PathManager(object):

    GRAPHS = 'graphs'
    GRAPHS_001_APPROX = 'graphs_001_approx'
    GRAPHS_005_APPROX = 'graphs_005_approx'

    ANALYSIS_001_APPROX = 'analysis_001_approx'
    ANALYSIS_005_APPROX = 'analysis_005_approx'
    MPEG7 = 'mpeg7'
    MNIST = 'mnist'
    FIGS = 'figs'

    def __init__(self, working_dir='tmp'):
        self.working_dir = working_dir

    def concat(self, *args):
        return os.path.join(*args)

    @property
    def data_dir(self):
        return self.concat(self.working_dir, 'data')

    @property
    def mpeg7_data_dir(self):
        return self.concat(self.data_dir, 'mpeg7')

    @property
    def mnist_data_dir(self):
        return self.concat(self.data_dir, 'mnist')

    @property
    def graphs_dir(self):
        return self.concat(self.working_dir, 'graphs')

    @property
    def rand_graphs(self):
        p = self.concat(self.graphs_dir, 'random')
        return GraphPath(p)

    @property
    def mpeg7_approx001_graphs(self):
        p = self.concat(self.graphs_dir, 'mpeg7-001')
        return GraphPath(p)



    #
    #
    #
    #     #
    #     # self._random_graphs_dir = 'graphs'
    #     # self._processed_graphs_dir = '???'
    #
    # def random_paths(self):
    #     exp_list = []
    #     for filename in os.listdir(os.path.join(self._random_graphs_dir,'random')):
    #         G = nx.read_gpickle(os.path.join(self._random_graphs_dir,'random' , filename))
    #         output_file = os.path.join("random", filename[:-8]+".txt")
    #         exp_list.append({"G":G, "output_file":output_file})
    #     return exp_list
    #
    # def mpeg_paths(self):
    #     exp_list = []
    #     for filename in os.listdir(os.path.join(self._processed_graphs_dir,'mpeg7')):
    #         G = nx.read_gpickle(os.path.join(self._processed_graphs_dir,'mpeg7', filename))
    #         output_file = os.path.join("mpeg7", filename[:-8]+".txt")
    #         exp_list.append({"G":G, "output_file":output_file})
    #     return exp_list
    #
    # def emnist_paths(self):
    #     exp_list = []
    #     for filename in os.listdir(os.path.join(self._processed_graphs_dir,'mnist')):
    #         #test_output_file = "mnist/"+filename[:-8]+".txt"
    #         #if not os.path.exists(out_graphs_dir+"/uniform_sample_exp/"+test_output_file):
    #         G = nx.read_gpickle(os.path.join(self._processed_graphs_dir,'mnist', filename))
    #         output_file = os.path.join("mnist", filename[:-8]+".txt")
    #         exp_list.append({"G":G, "output_file":output_file})
    #     return exp_list
    #
    #
    # def test_paths(self):
    #     exp_list = []
    #     #get one random graph
    #     G = nx.read_gpickle(self._random_graphs_dir + '/random/RAND_3_1.gpickle')
    #     output_file = "TEST_RAND.txt"
    #     exp_list.append({"G":G, "output_file":output_file})
    #
    #     #get the first MPEG7 file
    #     G1 = nx.read_gpickle(self._processed_graphs_dir + '/mpeg7/MPEG7_apple-1.gpickle')
    #     output_file = "TEST_MPEG7.txt"
    #     exp_list.append({"G":G1, "output_file":output_file})
    #
    #     #get the first MNIST file
    #     G2 = nx.read_gpickle(self._processed_graphs_dir + '/mnist/MNIST_C1_S0.gpickle')
    #     output_file = "TEST_MNIST.txt"
    #     exp_list.append({"G":G2, "output_file":output_file})
    #
    #     return exp_list
    #
    # def output_paths(self, output_dir):
    #     paths = []
    #     for exp in ['smallest_angle_exp', 'uniform_sample_exp']:
    #         for dataset in ['mnist', 'mpeg7', 'random']:
    #             paths.append(os.path.join(output_dir, experiment, dataset])
    #     return paths
    #
