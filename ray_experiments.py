import argparse
import logging
import math

import numpy as np
import tensorflow as tf
import ray
import ray.tune as tune

import model
import ray_workers
import tflog_utils
import utils
import ray_configs as configs

from ray_extensions import ExtendedTrainable

np.warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class LLFSExperiment(ExtendedTrainable):

    def _setup(self, config):
        self.target_timesteps = 1
        logger.warning('Starting experiment')

        if not isinstance(config['env_name'], list):
            config['env_name'] = [config['env_name']]
        self.dconfig = dconfig = utils.DotDict(config)

        self.summary_writer = self.find_tf_logger() or tf.summary.FileWriter(self.logdir)
        tflog_utils.log_text(self.summary_writer, 'config', str(dconfig))

        # Assign different environments to different agents
        env_count = len(config['env_name'])
        agent_configs = [utils.merge_dicts(config, {
            'env_name': config['env_name'][i % env_count]
        }) for i in range(dconfig.agent_count)]

        self.agents = [ray_workers.AgentWorker.remote(i, agent_configs[i], self.logdir) for i in range(dconfig.agent_count)]
        logger.warning('Setting up agents')
        # [ray] There is no way to wait for the actors to finalize initialization, thus we put this in a setup method
        ray.wait([agent.setup.remote() for agent in self.agents], num_returns=dconfig.agent_count)
        logger.warning('Created agents')

        if dconfig.restore_count:
            self._restore_from_specification(dconfig, agent_configs)

        # Create objective server and sync objective parameters
        if dconfig.agent_count > 1:
            params = self.agents[0].get_objective_params.remote()
            self.server = ray_workers.ObjectiveServer.remote(config, params)
            logger.warning('Created server')
            self.obj_param_count = len(ray.get(params))
            ray.wait([agent.update_objective_params.remote(params) for agent in self.agents[1:]],
                     num_returns=dconfig.agent_count - 1)
            logger.warning('Synced objective function')

    def _restore_from_specification(self, dconfig, agent_configs):
        """
        Restores policies, critics, and / or objective functions from checkpoints
        """
        env_count = len(dconfig.env_name)
        if dconfig.restore_count > 1:
            spec_restore_agents = dconfig.restore_agents
        else:
            spec_restore_agents = [dconfig.restore_agents]
        for i, (path, start, stop, num, restore_replay_buffer) in enumerate(spec_restore_agents):

            def get_restore_paths(base_path, start, stop, num):
                """
                Get `num` checkpoint paths from `base_path` at index `start` up until `stop`

                :param base_path: Tensorflow checkpoint path
                :param start: First zero-based checkpoint to use (int)
                :param stop: A float in the interval [0, 1] describing the maximum percentile until which checkpoints
                                are taken, i.e. 0.5 uses only the first 50% of all checkpoints
                :param num: The maximum number of checkpoints to use
                :return: A list of checkpoint paths
                """
                paths = tf.train.get_checkpoint_state(base_path).all_model_checkpoint_paths
                paths = np.array(paths)
                if stop > 0:
                    idxs = np.linspace(start, int(stop * len(paths)), num, dtype=np.int32, endpoint=False)
                else:
                    idxs = np.arange(start, start + num)
                return paths[idxs]

            env_restore_paths = {env: get_restore_paths(path, start, stop, num) for env in dconfig.env_name}
            ray.wait([agent.restore.remote(env_restore_paths[agent_config['env_name']][j // env_count],
                                           restore_saver=i, restore_history=restore_replay_buffer)
                      for j, (agent, agent_config) in enumerate(zip(self.agents, agent_configs))],
                     num_returns=dconfig.agent_count)
        logger.warning('Restored agents')

    def _train(self):
        """
        Run config.steps episodes of training, then return control to ray
        """

        timesteps_total = self._timesteps_total or 0
        timesteps_this_iter = 0
        t = timesteps_total
        reward_accumulator = []

        # Ray object id for the objective function parameters
        var_oid = None
        # Ray object ids for the objective function gradients of each agent
        grad_oids = [None for _ in range(self.dconfig.agent_count)]

        for _ in range(self.dconfig.steps):
            # Collect experience
            simulation_objs = [agent.simulate.remote(t, self.target_timesteps) for agent in self.agents]
            interaction_lengths, shortest_episodes, rewards = zip(*ray.get(simulation_objs))
            max_interaction_length = max(interaction_lengths)
            self.target_timesteps = max(shortest_episodes)
            timesteps_this_iter += max_interaction_length
            t = timesteps_total + timesteps_this_iter
            reward_accumulator.extend(rewards)

            # Update critics, policies, and objective function in parallel
            for j in range(max_interaction_length):
                should_update_policy = j % self.dconfig.policy_update_delay == 0
                should_update_objective = self.dconfig.obj_func_enabled \
                                          and self.dconfig.obj_func_update_delay != -1 \
                                          and j % self.dconfig.obj_func_update_delay == 0
                # Whether to update objective locally or sync gradients
                should_update_objective_grads = should_update_objective and self.dconfig.agent_count > 1
                should_update_objective_local = should_update_objective and self.dconfig.agent_count == 1

                if should_update_objective_grads:
                    grad_oids = [utils.plasma_create_id() for _ in range(self.dconfig.agent_count)]

                for idx, agent in enumerate(self.agents):
                    # Issue agent update commands remotely
                    agent.update.remote(t, critic=True, policy=should_update_policy,
                                        var_oid=var_oid, grad_oid=grad_oids[idx],
                                        objective_local=should_update_objective_local,
                                        objective_grads=should_update_objective_grads)

                if should_update_objective_grads:
                    var_oid = utils.plasma_create_id()
                    # Issue agent gradient merge and application remotely
                    self.server.apply_gradients.remote(grad_oids, var_oid)

        if self.dconfig.agent_count > 1:
            # Sync objective function parameters
            for agent in self.agents:
                agent.update_objective_params.remote(oid=var_oid)

        # Log to tensorboard and wait for all agents
        ray.wait([agent.write_summary.remote(t) for agent in self.agents], num_returns=self.dconfig.agent_count)

        # Return training status, will be logged to tensorboard by ray
        return {'timesteps_this_iter': timesteps_this_iter,
                'mean_reward': np.mean(reward_accumulator),
                'config': self.config}

    def _stop(self):
        self.summary_writer.close()
        ray.wait([agent.stop.remote() for agent in self.agents], num_returns=self.dconfig.agent_count)
        del self.agents
        if self.dconfig.agent_count > 1:
            del self.server

    def _save(self, checkpoint_dir):
        prefixes = ray.get([agent.save.remote(f'{checkpoint_dir}/agent_{i}', self._timesteps_total)
                            for i, agent in enumerate(self.agents)])
        return {"prefixes": prefixes}

    def _restore(self, checkpoint_data):
        prefixes = checkpoint_data["prefixes"]
        ray.wait([agent.restore.remote(prefix) for agent, prefix in zip(self.agents, prefixes)],
                 num_returns=self.dconfig.agent_count)


# noinspection PyProtectedMember
def count_required_gpus(config):
    if config['agent_count'] > 1:
        return math.ceil(config['agent_count'] * ray_workers.AgentWorker._num_gpus + ray_workers.ObjectiveServer._num_gpus)
    else:
        return ray_workers.AgentWorker._num_gpus


def init_ray(redis_address=None):
    if redis_address:
        ray.init(redis_address=redis_address)
    else:
        mem = 1000 * 1000 * 1000  # 1 GB
        ray.init(object_store_memory=mem, redis_max_memory=mem, temp_dir='/tmp/metagenrl/ray')


def run(config, run_name='metagenrl', timesteps=300 * 1000, samples=1):
    tune.register_trainable(run_name, LLFSExperiment)
    trial_gpus = count_required_gpus(config)
    print(f'Requiring {trial_gpus} extra gpus.')
    train_spec = {
        'run': run_name,
        'resources_per_trial': {'cpu': 0, 'gpu': 0, 'extra_gpu': trial_gpus},
        'stop': {'timesteps_total': timesteps},
        'config': config,
        'num_samples': samples,
        'checkpoint_at_end': True,
    }
    tune.run_experiments({'metagenrl': train_spec})


def train(args):
    """
    Performs meta-training
    """
    config = configs.base(agent_count=20)
    config.update({
        'env_name': [
            'LunarLanderContinuous-v2',
            'HalfCheetah-v2',
        ],
    })

    run(config, run_name='public-CheetahLunar')


def test(args):
    """
    Performs meta-test training
    """
    assert isinstance(args.objective, str)
    config = configs.test(args.objective)
    config.update({
        'env_name': tune.grid_search([
            'Hopper-v2',
        ]),
    })

    run(config, run_name='test-public-CheetahLunar')


if __name__ == '__main__':
    FUNCTION_MAP = {'train': train,
                    'test': test}

    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=FUNCTION_MAP.keys())
    parser.add_argument('--redis', dest='redis_address', action='store', type=str)
    parser.add_argument('--objective', action='store', type=str)
    parsed_args = parser.parse_args()
    init_ray(parsed_args.redis_address)
    func = FUNCTION_MAP[parsed_args.command]
    func(parsed_args)
