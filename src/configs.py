import numpy as np

class Configs():

    def __init__(self, dict):
        self.pop_size = dict['pop_size']
        self.elitism_ratio = dict['elitism_ratio']
        self.cross_rate = dict['cross_rate']
        self.cross_method = dict['cross_method']
        self.input_size = dict['input_size']
        self.output_size = dict['output_size']
        self.weight_mean = dict['weight_mean']
        self.weight_std = dict['weight_std']
        self.perturb_std = dict['perturb_std']
        self.weight_perturb_rate = dict['weight_perturb_rate']
        if 'weight_zero_rate' in dict:
            self.weight_zero_rate = dict['weight_zero_rate']
        else:
            self.weight_zero_rate = 0

        self.encoding = dict['encoding']
        if self.encoding == "MENNAG":
            self.max_depth = dict['max_depth']
            self.feedforward = dict['feedforward']
            self.forward_prob = dict['forward_prob']
            self.conn_relocate_rate = dict['conn_relocate_rate']
            self.weight_reset_rate = dict['weight_reset_rate']
            self.bias_rate = dict['bias_rate']
            self.insertion_rate = dict['insertion_rate']
            self.deletion_rate = dict['deletion_rate']
            self.random_tree_rate = dict['random_tree_rate']
        elif self.encoding == "direct":
            self.layer_size = dict['layer_size']
            self.all_size = 0
            for i in range(0, len(self.layer_size) - 1):
                self.all_size += (self.layer_size[i] + 1) * self.layer_size[i + 1]

        if self.cross_method in ["modular", "modular2", "global_modular", "op"]:
            self.cross_module_number = dict['cross_module_number']
        elif self.cross_method == "jagged":
            self.cross_module_number = dict['cross_module_number']
            self.module_identifier = np.random.randint(self.cross_module_number, size = self.all_size)

        self.ea = dict['ea']
        if self.ea == "speciation":
            self.speciation_range = dict['speciation_range']
            self.crossover_range = dict['crossover_range']
            self.random_crossover_rate = dict['random_crossover_rate']
            self.max_ancestry_dist = max(self.speciation_range, self.crossover_range[1])
            try:
                self.fitness_sharing = dict['fitness_sharing']
            except KeyError:
                self.fitness_sharing = False
        if self.ea == "mapelite":
            self.initial_batch = dict['initial_batch']
            self.batch_size = dict['batch_size']
            self.feature_range = dict['feature_range']
            self.resolution = dict['resolution']

        if self.ea == "optimal_mixing":
            self.FOS = dict['FOS']
            if self.FOS:
                self.FOS_type = dict['FOS_type']
            self.OM_random_accept_rate = dict['OM_random_accept_rate']
            self.OM_type = dict['OM_type']

        if 'max_evals' in dict:
            self.max_evals = dict['max_evals']

        self.symmetry_rate = dict['symmetry_rate']
        if self.symmetry_rate > 0:
            self.input_cayley = dict['input_cayley']
            self.output_cayley = dict['output_cayley']
        if 'metric' in dict:
            self.metric = dict['metric']
        else:
            self.metric = 'fit'
        if 'avail_acts' in dict:
            self.avail_acts = dict['avail_acts']
        else:
            self.avail_acts = [1]

        if 'behavior_similarity' in dict:
            self.behavior_similarity = dict['behavior_similarity']
            self.behavior_similarity_measure = dict['behavior_similarity_measure']
        else:
            self.behavior_similarity = False

        if 'phenotype_similarity' in dict:
            self.phenotype_similarity = dict['phenotype_similarity']
            self.phenotype_similarity_measure = dict['phenotype_similarity_measure']
        else:
            self.phenotype_similarity = False

        if 'neuron_similarity' in dict:
            self.neuron_similarity = dict['neuron_similarity']
        else:
            self.neuron_similarity = False

        if 'ns_interval' in dict:
            self.ns_interval = dict['ns_interval']
        else:
            self.ns_interval = 1

        self.acts_mut_rate = dict['acts_mut_rate']

        if 'l1' in dict:
            self.l1 = dict['l1']
            self.l1_coef = dict['l1_coef']
        else:
            self.l1 = False
        if 'l0' in dict:
            self.l0 = dict['l0']
            self.l0_coef = dict['l0_coef']
            self.l0_threshold = dict['l0_threshold']
        else:
            self.l0 = False

        try:
            self.cross_norm = dict['cross_norm']
        except KeyError:
            self.cross_norm = False

        self.batch_size = 1
        if 'batch_size' in dict:
            self.batch_size = dict['batch_size']