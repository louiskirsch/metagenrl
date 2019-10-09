import os
import ray.tune as tune

checkpoint_base_path = os.path.expanduser('~/ray_results/metagenrl/')
objective_dir_postfix = '/checkpoint/agent_0'


def test(objective_dir='', chkp=-1):
    """
    Test the given objective function by training new agents from scratch

    :param objective_dir: the name of the run directory where the objective function is saved,
                            can be a list to test multiple
    :param chkp: The checkpoint index to test, can be a list to test multiple
    """
    if isinstance(objective_dir, str) and isinstance(chkp, int):
        objective_dir = checkpoint_base_path + objective_dir + objective_dir_postfix
        restore_agents = (objective_dir, chkp, 0, 1, False)
    elif isinstance(objective_dir, list) and isinstance(chkp, int):
        objective_dirs = [checkpoint_base_path + p + objective_dir_postfix for p in objective_dir]
        restore_agents = tune.grid_search([(d, chkp, 0, 1, False)
                                           for d in objective_dirs])
    elif isinstance(chkp, list):
        objective_dir = checkpoint_base_path + objective_dir + objective_dir_postfix
        restore_agents = tune.grid_search([(objective_dir, pos, 0, 1, False)
                                           for pos in chkp])
    else:
        raise ValueError('Invalid arguments', objective_dir, chkp)

    config = base(agent_count=1)
    config.update({
        'env_name': 'LunarLanderContinuous-v2',
        'restore_count': 1,
        'restore': ['objective'],
        'restore_agents': restore_agents,
        'obj_func_update_delay': -1,
        'obj_func_anneal_steps': 0,
        'policy_update_start': 0,
        'policy_random_exploration_steps': 10 * 1000,

    })
    return config


def reinforce_test():
    """
    Use a fixed objective function resembling off-policy REINFORCE with GAE
    """
    config = base(agent_count=1)
    config.update({
        'env_name': 'LunarLanderContinuous-v2',
        'restore_count': 0,
        'restore': None,
        'restore_agents': None,
        'obj_func_type': 'reinforce',
        'obj_func_update_delay': -1,
        'obj_func_anneal_steps': 0,
        'policy_update_start': 0,
        'policy_random_exploration_steps': 10 * 1000,
    })
    return config


def baseline():
    """
    Update the policy directly with the critic, i.e. run DDPG
    """
    config = base(agent_count=1)
    config.update({
        'policy_update_start': 0,
        'obj_func_anneal_steps': 0,
        'policy_random_exploration_steps': 10 * 1000,
        'restore_count': 0,
        'restore': None,
        'restore_agents': None,
        'obj_func_enabled': False,
        'obj_func_type': 'reinforce',  # This will be ignored
    })
    return config


def base(agent_count=8):
    """
    Return config with default parameters
    """
    return {
        'env_name': 'LunarLanderContinuous-v2',
        'max_episode_length': None,
        'steps': 10,
        'clip_gradient': 1.0,
        'recurrent_time_steps': 20,
        'agent_count': agent_count,
        'restore_count': 0,
        'restore': None,
        'restore_agents': None,

        'critic_depth': 3,
        'critic_units': 350,
        'critic_activation': 'relu',
        'critic_rnn_activation': 'tanh',
        'critic_layernorm': True,
        'critic_is_recurrent': False,
        'critic_learning_rate': 1e-3,
        'critic_noise': 0.2,
        'critic_noise_clip': 0.5,
        'target_network_update_speed': 1 - 0.995,
        'discount_factor': 0.99,
        'gae_factor': 0.97,
        'buffer_sample_size': 100,
        'buffer_size': 1000000,

        'policy_depth': 3,
        'policy_units': 350,
        'policy_activation': 'relu',
        'policy_rnn_activation': 'tanh',
        'policy_layernorm': True,
        'policy_is_recurrent': False,
        'policy_learning_rate': 1e-3,
        'policy_lr_annealing_base': 25,
        'policy_update_delay': 2,
        'policy_update_start': 0,
        'policy_clip': True,
        'policy_exploration': 0.1,
        'policy_reset_prob': 0,
        'policy_random_exploration_steps': 10 * 1000,

        'obj_func_enabled': True,
        'obj_func_type': 'learned-reinforce',
        'obj_func_depth': 3,
        'obj_func_units': 32,
        'obj_func_lstm_units': 32,
        'obj_func_input_transform_depth': 3,
        'obj_func_input_transform_units': 32,
        'obj_func_input_transform_out_units': 8,
        'obj_func_input_transform_layernorm': False,
        'obj_func_activation': ['relu', 'square'],
        'obj_func_layernorm': True,
        'obj_func_learning_rate': 1e-3,
        'obj_func_error_scale': 1e-3,
        'obj_func_error_func': 'tanh',
        'obj_func_second_order_adam': False,
        'obj_func_second_order_stepsize': 1e-3,
        'obj_func_second_order_steps': 1,
        'obj_func_update_delay': 2,
        'obj_func_anneal_steps': 10 * 1000,
    }
