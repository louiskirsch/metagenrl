import os

import ray
from unittest import TestCase

import ray_experiments


class TestExperiment(TestCase):

    SIMPLE_CONFIG = {
        'env_name': 'LunarLanderContinuous-v2',
        'max_episode_length': 1000,
        'steps': 2,
        'clip_gradient': 1.0,
        'recurrent_time_steps': 3,  # set to 1 if not recurrent
        'restore': ['!objective', '!agent/trained'],
        'agent_count': 1,

        'critic_depth': 2,
        'critic_units': 5,
        'critic_activation': 'relu',
        'critic_rnn_activation': 'tanh',
        'critic_layernorm': False,
        'critic_is_recurrent': False,
        'critic_learning_rate': 1e-3,
        'critic_noise': 0.2,
        'critic_noise_clip': 0.5,
        'target_network_update_speed': 1 - 0.995,
        'discount_factor': 0.99,
        'gae_factor': 0.97,
        'buffer_sample_size': 100,
        'buffer_size': 1000000,

        'policy_depth': 2,
        'policy_units': 5,
        'policy_activation': 'relu',
        'policy_rnn_activation': 'tanh',
        'policy_layernorm': False,
        'policy_is_recurrent': False,
        'policy_learning_rate': 1e-3,
        'policy_update_delay': 2,
        'policy_clip': True,
        'policy_exploration': 0.1,
        'policy_reset_prob': 0,
        'policy_random_exploration_steps': 2,

        'obj_func_enabled': False,
        'obj_func_type': 'learned-reinforce',
        'obj_func_depth': 2,
        'obj_func_units': 32,
        'obj_func_lstm_units': 8,
        'obj_func_input_transform_depth': 2,
        'obj_func_input_transform_units': 8,
        'obj_func_input_transform_out_units': 8,
        'obj_func_input_transform_layernorm': False,
        'obj_func_activation': 'relu',
        'obj_func_layernorm': False,
        'obj_func_learning_rate': 1e-3,
        'obj_func_second_order_stepsize': 1e-4,
        'obj_func_second_order_steps': 1,
        'obj_func_update_delay': 2,
        'obj_func_anneal_steps': None,
    }

    def setUp(self):
        ray.init(local_mode=True)

    def tearDown(self):
        ray.shutdown()

    def _train(self, config):
        experiment = ray_experiments.LLFSExperiment(config)
        experiment.train()
        experiment.stop()

    def test__train(self):
        self._train(self.SIMPLE_CONFIG)

    def test__objective_function_reinforce(self):
        config = self.SIMPLE_CONFIG.copy()
        config['obj_func_enabled'] = True
        config['obj_func_type'] = 'reinforce'
        self._train(config)

    def test__objective_function_learned_reinforce_backwards_rnn(self):
        config = self.SIMPLE_CONFIG.copy()
        config['obj_func_enabled'] = True
        config['obj_func_type'] = 'learned-reinforce'
        self._train(config)

    def test__objective_function_mixed_activations(self):
        config = self.SIMPLE_CONFIG.copy()
        config['obj_func_enabled'] = True
        config['obj_func_type'] = 'learned-reinforce'
        config['obj_func_activation'] = ['relu', 'square']
        self._train(config)

    def test__objective_function_annealing(self):
        config = self.SIMPLE_CONFIG.copy()
        config['obj_func_enabled'] = True
        config['obj_func_anneal_steps'] = 5
        self._train(config)

    def test__objective_function_multiple_steps(self):
        config = self.SIMPLE_CONFIG.copy()
        config['obj_func_enabled'] = True
        config['obj_func_second_order_steps'] = 4
        config['obj_func_second_order_adam'] = True
        self._train(config)

    def test__objective_function_policy_reset(self):
        config = self.SIMPLE_CONFIG.copy()
        config['obj_func_enabled'] = True
        config['policy_reset_prob'] = 1e-1
        self._train(config)

    def test__layernorm(self):
        config = self.SIMPLE_CONFIG.copy()
        config['obj_func_enabled'] = True
        config['critic_layernorm'] = True
        config['policy_layernorm'] = True
        config['obj_func_layernorm'] = True
        self._train(config)

    def test__restore(self):
        experiment = ray_experiments.LLFSExperiment(self.SIMPLE_CONFIG)
        path = experiment.save()
        experiment.restore(path)


class TestMultiAgentExperiment(TestCase):

    @classmethod
    def setUpClass(cls):
        ray.init()

    def _train(self, config):
        experiment = ray_experiments.LLFSExperiment(config)
        experiment.train()
        experiment.stop()

    def test__objective_function_multi_agent(self):
        config = TestExperiment.SIMPLE_CONFIG.copy()
        config['obj_func_enabled'] = True
        config['agent_count'] = 3
        self._train(config)

    def test__objective_function_multi_agent_deep(self):
        config = TestExperiment.SIMPLE_CONFIG.copy()
        config['obj_func_enabled'] = True
        config['agent_count'] = 2
        config['obj_func_depth'] = 4
        config['obj_func_units'] = 256

