import pickle

import gym
import numpy as np
import ray
import tensorflow as tf
from pyarrow import plasma as plasma

import tflog_utils
import utils
from model import logger, Agent, Objective, ReplayBuffer
from utils import placeholder, get_vars


@ray.remote(num_cpus=0, num_gpus=0.001)
class ObjectiveServer:
    """
    A ray worker receiving the gradients from the agents and sending back the new objective function parameters
    """

    def __init__(self, config, init_vars):
        dconfig = utils.DotDict(config)

        import tensorflow as tf
        plasma.load_plasma_tensorflow_op()

        store_socket = utils.get_store_socket()
        self.var_oid = None

        self.obj_vars = [tf.Variable(init_var, name='obj_var', dtype=tf.float32)
                         for init_var in init_vars]
        self.plasma_grads_oids = tf.placeholder(shape=[dconfig.agent_count],
                                                dtype=tf.string, name="plasma_grads_oids")
        self.plasma_vars_oid = tf.placeholder(shape=[],
                                              dtype=tf.string, name="plasma_vars_oids")

        shapes = [v.shape for v in self.obj_vars]
        grads = utils.reverse_flat(tf.reduce_mean(
            [plasma.tf_plasma_op.plasma_to_tensor(self.plasma_grads_oids[a], dtype=tf.float32,
                                                  plasma_store_socket_name=store_socket)
             for a in range(dconfig.agent_count)], axis=0), shapes)

        obj_optimizer = tf.train.AdamOptimizer(learning_rate=dconfig.obj_func_learning_rate)
        self.train_obj_op = obj_optimizer.apply_gradients(zip(grads, self.obj_vars))
        with tf.control_dependencies([self.train_obj_op]):
            self.update_vars = plasma.tf_plasma_op.tensor_to_plasma([utils.flat(self.obj_vars)],
                                                                    self.plasma_vars_oid,
                                                                    plasma_store_socket_name=store_socket)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())

    def apply_gradients(self, grad_oids, var_oid):
        utils.plasma_prefetch(grad_oids)
        feed_dict = {
            self.plasma_grads_oids: grad_oids,
            self.plasma_vars_oid: var_oid
        }
        self.sess.run(self.update_vars, feed_dict)

        # Free resources
        if self.var_oid is not None:
            utils.plasma_free([self.var_oid])
        self.var_oid = var_oid
        utils.plasma_free(grad_oids)


# Use five workers per GPU
# [ray] Due to a bug with floating point resources we need to subtract a small epsilon
# Also allows ObjectiveServer to be on the same GPU
@ray.remote(num_cpus=0, num_gpus=0.2 - 0.001)
class AgentWorker:
    """
    A ray worker that represents an agent with replay buffer, critic, and policy
    """

    def __init__(self, worker_index, config, logdir):
        logger.warning(f'Create agent {worker_index}')
        self.dconfig = utils.DotDict(config)
        self.logdir = logdir
        self.worker_index = worker_index
        self.locals = None
        self.feed_dict = None
        self.objective_vars_oid = None
        self.datasets_initialized = False

        import tensorflow as tf
        plasma.load_plasma_tensorflow_op()

        logger.warning(f'Created agent {worker_index}')

    def setup(self):
        logger.warning(f'Setting up agent {self.worker_index}')
        tf.reset_default_graph()
        self.locals = self._setup(self.dconfig, self.logdir)

    def _setup(self, dconfig, logdir):
        """
        Create tensorflow graph and summary writer
        :param dconfig: configuration to use to build the graph
        :param logdir: log directory to write tensorflow logs to
        """
        env = gym.make(dconfig.env_name)
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]

        # Action limit for clamping: critically, assumes all dimensions share the same bound!
        act_limit = env.action_space.high[0]

        agent = Agent(dconfig, env)
        objective = Objective(dconfig)

        # Experience buffer
        replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=dconfig.buffer_size,
                                     discount_factor=dconfig.discount_factor)

        time = dconfig.recurrent_time_steps if dconfig.recurrent_time_steps > 1 else None

        # Create datasets from replay buffer
        replay_buffer_dataset = replay_buffer.create_dataset(dconfig.buffer_sample_size, time)
        replay_buffer_dataset_iterator = replay_buffer_dataset.make_initializable_iterator()

        # If we perform multiple gradient steps in the inner loop, provide different data for each step
        large_batch_size = (self.dconfig.obj_func_second_order_steps + 1) * dconfig.buffer_sample_size
        large_replay_buffer_dataset = replay_buffer.create_dataset(large_batch_size, time)
        large_replay_buffer_dataset_iterator = large_replay_buffer_dataset.make_initializable_iterator()

        handle = tf.placeholder(tf.string, shape=[])
        iterator = tf.data.Iterator.from_string_handle(handle, replay_buffer_dataset.output_types,
                                                       replay_buffer_dataset.output_shapes)
        itr_elem = utils.DotDict(iterator.get_next())
        x_ph, a_ph, x2_ph, r_ph, d_ph, lens_ph = itr_elem.obs1, itr_elem.acts, itr_elem.obs2,\
                                                 itr_elem.rews, itr_elem.done, itr_elem.lens

        # Mask for different trajectory lengths
        if lens_ph is not None:
            seq_mask = tf.sequence_mask(lens_ph, time, dtype=tf.float32)
        else:
            seq_mask = tf.ones([], dtype=tf.float32)

        x_ph_behv = placeholder(obs_dim, name='ObsBehavior')
        timestep = tf.placeholder(tf.float32, [], 'timestep')

        if dconfig.policy_is_recurrent:
            state_shape = [2, 1, dconfig.policy_units]
            init_policy_state = tf.placeholder_with_default(tf.zeros(state_shape), [2, 1, dconfig.policy_units])
        else:
            init_policy_state = None

        transition = [x_ph, a_ph, x2_ph, r_ph[..., tf.newaxis], d_ph[..., tf.newaxis]]

        # Learning rate annealing
        if dconfig.policy_update_start:
            base = dconfig.policy_lr_annealing_base
            lr_progress = (base ** tf.minimum(1.0, timestep / dconfig.policy_update_start) - 1) / (base - 1)
        else:
            lr_progress = 1

        # Optimizers
        pi_optimizer = utils.TensorAdamOptimizer(learning_rate=dconfig.policy_learning_rate * lr_progress)
        q_optimizer = tf.train.AdamOptimizer(learning_rate=dconfig.critic_learning_rate)
        obj_optimizer = tf.train.AdamOptimizer(learning_rate=dconfig.obj_func_learning_rate)

        # Main outputs from computation graph
        main = agent.main
        policy = main.policy(x_ph, seq_len=lens_ph)
        pi_action = policy.action
        q1_pi = policy.value
        pi_behv = main.policy(x_ph_behv[:, tf.newaxis], initial_state=init_policy_state)
        q1 = main.critic(x_ph, a_ph)
        q2 = main.critic2(x_ph, a_ph)
        obj = objective.objective(x_ph, a_ph, transition, lens_ph, seq_mask, agent, policy)

        # Target policy network
        pi_action_targ = agent.target.policy(x2_ph, seq_len=lens_ph).action

        # Target Q networks
        # Target policy smoothing, by adding clipped noise to target actions
        epsilon = tf.random_normal(tf.shape(pi_action_targ), stddev=dconfig.critic_noise)
        epsilon = tf.clip_by_value(epsilon, -dconfig.critic_noise_clip, dconfig.critic_noise_clip)
        a2 = pi_action_targ + epsilon
        a2 = tf.clip_by_value(a2, -act_limit, act_limit)
        q1_targ = agent.target.critic(x2_ph, a2)
        q2_targ = agent.target.critic2(x2_ph, a2)

        # Bellman backup for Q functions, using Clipped Double-Q targets
        min_q_targ = tf.minimum(q1_targ, q2_targ)
        gamma = dconfig.discount_factor
        backup = tf.stop_gradient(r_ph + gamma * (1 - d_ph) * min_q_targ + d_ph)

        # Objective function annealing
        if dconfig.obj_func_anneal_steps:
            progress = tf.minimum(1.0, timestep / dconfig.obj_func_anneal_steps)
            obj = progress * obj - (1 - progress) * q1_pi

        # TD3 losses
        pi_loss = -tf.reduce_mean(q1_pi * seq_mask)
        pi_obj_loss = tf.reduce_mean(obj * seq_mask)
        q1_loss = tf.reduce_mean((q1-backup)**2 * seq_mask)
        q2_loss = tf.reduce_mean((q2-backup)**2 * seq_mask)
        q_loss = q1_loss + q2_loss

        main_vars = sorted(get_vars('main', trainable_only=False), key=lambda v: v.name)
        target_vars = sorted(get_vars('target', trainable_only=False), key=lambda v: v.name)

        # Train policy directly using critic
        train_pi_op = self._clipped_minimize(pi_optimizer, pi_loss, get_vars('main/policy'),
                                             grad_name='ddpg_policy_grads')
        # Train policy using objective function
        train_pi_obj_op = self._clipped_minimize(pi_optimizer, pi_obj_loss, get_vars('main/policy'),
                                                 grad_name='objective_policy_grads')
        # Train critic
        train_q_op = q_optimizer.minimize(q_loss, var_list=get_vars('main/critic'))
        tf.summary.histogram('policy_params', utils.flat(get_vars('main/policy')))

        # Objective function loss
        q1_obj = objective.future_policy_value(x_ph, a_ph, transition, lens_ph, seq_mask, agent, pi_optimizer,
                                               create_summary=dconfig.obj_func_enabled)
        obj_loss = -tf.reduce_mean(q1_obj)

        # Objective function optimization using ray (send gradients to ObjectiveServer)
        obj_vars = get_vars('objective')
        store_socket = utils.get_store_socket()

        shapes = [v.shape for v in obj_vars]
        plasma_var_oid = tf.placeholder(shape=[], dtype=tf.string, name="plasma_var_oid")
        retrieved_vars = utils.reverse_flat(plasma.tf_plasma_op.plasma_to_tensor(plasma_var_oid, dtype=tf.float32,
                                                                                 plasma_store_socket_name=store_socket),
                                            shapes)
        # Op to read new objective parameters from ray object store
        plasma_read_vars = [var.assign(retrieved) for var, retrieved in zip(obj_vars, retrieved_vars)]

        grads, vars = zip(*obj_optimizer.compute_gradients(obj_loss, obj_vars))
        grads, _ = tf.clip_by_global_norm(grads, clip_norm=dconfig.clip_gradient)
        tf.summary.histogram('objective_params', utils.flat(vars))
        tf.summary.histogram('objective_param_grads', utils.flat(grads))
        objective_grads = grads
        # Op to send gradients to ObjectiveServer
        train_obj_op = obj_optimizer.apply_gradients(zip(objective_grads, vars))

        plasma_grad_oid = tf.placeholder(shape=[], dtype=tf.string, name="plasma_grad_oid")
        # Op to send gradients to ObjectiveServer
        plasma_write_grads = plasma.tf_plasma_op.tensor_to_plasma([utils.flat(objective_grads)],
                                                                  plasma_grad_oid,
                                                                  plasma_store_socket_name=store_socket)

        # Print number of parameters
        print(f'''
        ===================================================================
        Parameters
        Policy {np.sum(np.prod(v.shape) for v in get_vars('main/policy'))}
        Critic {np.sum(np.prod(v.shape) for v in get_vars('main/critic'))}
        Objective {np.sum(np.prod(v.shape) for v in obj_vars)}
        ===================================================================
        ''')

        # Polyak averaging for target variables
        polyak = 1 - dconfig.target_network_update_speed
        target_update = tf.group([tf.assign(v_targ, polyak*v_targ + (1-polyak)*v_main)
                                  for v_main, v_targ in zip(main_vars, target_vars)])

        # Initializing target networks to match main variables
        target_init = tf.group([tf.assign(v_targ, v_main)
                                for v_main, v_targ in zip(main_vars, target_vars)])

        # Ops for copying and resetting the policy (currently not used)
        reset_policy = tf.variables_initializer(get_vars('main'))
        copy_policy = tf.group([tf.assign(v_targ, v_main)
                                for v_main, v_targ in zip(get_vars('main'),
                                                          get_vars('target'))])

        # Summaries
        tflog_utils.log_scalars(policy_loss=pi_loss, q_loss=q_loss)
        if dconfig.obj_func_enabled:
            tflog_utils.log_scalars(policy_obj_loss=pi_obj_loss, objective_loss=obj_loss)

        self.restore_savers = self._create_restore_savers(dconfig)
        self.saver = tf.train.Saver(max_to_keep=1000, save_relative_paths=True)
        self.summary = tf.summary.merge_all()
        self.summary_writer = tf.summary.FileWriter(f'{logdir}_agent{self.worker_index}')

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())

        init_ops = [target_init]
        self.sess.run(init_ops)

        rb_handle, large_rb_handle = self.sess.run([replay_buffer_dataset_iterator.string_handle(),
                                                    large_replay_buffer_dataset_iterator.string_handle()])

        # Return all created tf ops
        return utils.DotDict(locals())

    def _create_restore_savers(self, dconfig):
        """
        Creates a saver used for restoring particular variables
        """
        if not dconfig.restore_count:
            return None

        restore_desc_list = dconfig.restore if dconfig.restore_count > 1 else [dconfig.restore]

        def restore_condition(desc):
            use_not = 0
            if desc[0] == '!':
                desc = desc[1:]
                use_not = 1
            return lambda vname: use_not ^ vname.startswith(desc)

        restore_conditions_list = [[restore_condition(cond) for cond in restore_desc]
                                   for restore_desc in restore_desc_list]
        vars_to_restore_list = [[v for v in tf.global_variables() if all([cond(v.name) for cond in restore_conditions])]
                           for restore_conditions in restore_conditions_list]
        restore_desc = [[(v.name, v.shape.as_list()) for v in vars_to_restore]
                        for vars_to_restore in vars_to_restore_list]
        print(f'Restoring: {restore_desc}')
        return [tf.train.Saver(vars_to_restore, save_relative_paths=True)
                for vars_to_restore in vars_to_restore_list]

    def simulate(self, timesteps_total, target_timesteps):
        """
        Interact with the environment for at least `target_timesteps`
        :param timesteps_total: How many timesteps already have passed since the beginning of training
        :param target_timesteps: How many additional timesteps to simulate (or more if episode not yet finished)
        """
        def get_action(o, noise_scale, state=None):
            """
            Generate a new action using the policy
            """
            if self.dconfig.policy_is_recurrent:
                behv_feed_dict = {self.locals.x_ph_behv: o.reshape(1, -1)}
                if state is not None:
                    behv_feed_dict[self.locals.init_policy_state] = state
                a, state = self.sess.run([self.locals.pi_behv.action, self.locals.pi_behv.state], behv_feed_dict)
            else:
                behv_feed_dict = {self.locals.x_ph_behv: o.reshape(1, -1)}
                a = self.sess.run(self.locals.pi_behv.action, behv_feed_dict)
            a = np.squeeze(a)
            a += noise_scale * np.random.randn(*a.shape)
            a = np.clip(a, -self.locals.act_limit, self.locals.act_limit)
            return np.asarray(a).reshape(self.locals.env.action_space.shape), state

        def simulate_episode():
            env = self.locals.env
            obs, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
            state = None
            taken_actions = []

            for _ in range(self.dconfig.max_episode_length or 10 ** 10):
                t = ep_len + timesteps_total
                start_steps = self.dconfig.policy_random_exploration_steps
                if t > start_steps:
                    a, state = get_action(obs, self.dconfig.policy_exploration, state)
                else:
                    a = env.action_space.sample()

                # Step the env
                new_obs, r, d, _ = env.step(a)
                taken_actions.append(a)
                ep_ret += r
                ep_len += 1

                d = ep_len == self.dconfig.max_episode_length or d

                # Store experience to replay buffer
                self.locals.replay_buffer.store(obs, a, r, new_obs, d)

                obs = new_obs

                if d:
                    break

            return ep_len, ep_ret, taken_actions

        timesteps = 0
        reward_total = 0
        episodes = 0
        shortest_episode = 1 << 16  # Large integer value
        while timesteps < target_timesteps:
            ep_len, episode_reward, taken_actions = simulate_episode()
            timesteps_total += ep_len
            timesteps += ep_len
            episodes += 1
            reward_total += episode_reward
            shortest_episode = min(shortest_episode, ep_len)

            summary = tf.Summary()
            tflog_utils.log_histogram(summary, 'distr_episode_actions', taken_actions)
            tflog_utils.log_scalar(summary, 'episode_reward', episode_reward)
            tflog_utils.log_scalar(summary, 'episode_length', ep_len)
            if timesteps >= target_timesteps:
                tflog_utils.log_scalar(summary, 'timesteps', timesteps)
            self.summary_writer.add_summary(summary, timesteps_total)

        # Reset policy (currently not used)
        if self.dconfig.policy_reset_prob:
            reset_prob = timesteps * self.dconfig.policy_reset_prob
            if np.random.random() <= reset_prob:
                self.sess.run(self.locals.reset_policy)
                self.sess.run(self.locals.copy_policy)

        return timesteps, shortest_episode, reward_total / episodes

    def update_critic(self, t):
        self.feed_dict = self._generate_feed_dict(t, self.locals.rb_handle)
        q_step_ops = [self.locals.train_q_op]
        self.sess.run(q_step_ops, self.feed_dict)

    def update_policy(self):
        if self.dconfig.obj_func_enabled:
            policy_op = self.locals.train_pi_obj_op
        else:
            policy_op = self.locals.train_pi_op

        ops = [policy_op, self.locals.target_update]

        self.sess.run(ops, self.feed_dict)

    def update(self, t, var_oid=None, grad_oid=None, critic=False, policy=False,
               objective_local=False, objective_grads=False):
        """
        Update the agent: critic, policy, and / or objective
        :param t: current time step
        :param var_oid: ray object id for objective parameters
        :param grad_oid: ray object id for objective gradients
        :param critic: whether to update the critic
        :param policy: whether to udpate the policy
        :param objective_local:  whether to update the objective locally
        :param objective_grads:  whether to compute gradients for the objective to update globally
        """
        if var_oid is not None:
            self.objective_vars_oid = var_oid
            utils.plasma_prefetch([var_oid])
        # TODO merge critic, policy, and objective OPs to single graph call?
        if critic:
            self.update_critic(t)
        if policy:
            self.update_policy()
        if objective_local:
            self.local_update_objective()
        if objective_grads:
            return self.compute_objective_gradients(t, grad_oid)

    def write_summary(self, t):
        self.feed_dict = self._generate_feed_dict(t, self.locals.large_rb_handle)
        summary = self.sess.run(self.summary, self.feed_dict)
        self.summary_writer.add_summary(summary, t)

    def _generate_feed_dict(self, t, rb_iterator_handle):
        self.ensure_init_datasets()
        feed_dict = {self.locals.timestep: t,
                     self.locals.handle: rb_iterator_handle}
        return feed_dict

    def stop(self):
        self.sess.close()
        self.summary_writer.close()

    def ensure_init_datasets(self):
        if self.datasets_initialized:
            return
        self.sess.run([self.locals.replay_buffer_dataset_iterator.initializer,
                       self.locals.large_replay_buffer_dataset_iterator.initializer])
        self.datasets_initialized = True

    def save(self, checkpoint_dir, global_step):
        path = checkpoint_dir + '/save'
        out = self.saver.save(self.sess, path, global_step=global_step, write_meta_graph=False)
        with open(out + '.history', mode='wb') as file:
            pickle.dump(self.locals.replay_buffer, file)
        return out

    def restore(self, checkpoint_path, restore_saver=-1, restore_history=True):
        saver = self.restore_savers[restore_saver] if restore_saver > -1 else self.saver
        if restore_history:
            with open(checkpoint_path + '.history', mode='rb') as file:
                self.locals.replay_buffer.restore(pickle.load(file))
        saver.restore(self.sess, checkpoint_path)

    def _clipped_minimize(self, optimizer, loss, vars, grad_name=None):
        grads, _ = zip(*optimizer.compute_gradients(loss, vars))
        grads, _ = tf.clip_by_global_norm(grads, clip_norm=self.dconfig.clip_gradient)
        if grad_name is not None:
            tf.summary.histogram(grad_name, utils.flat(grads))
        return optimizer.apply_gradients(zip(grads, vars))

    def local_update_objective(self):
        self.sess.run(self.locals.train_obj_op, self.feed_dict)

    def get_objective_params(self):
        return self.sess.run(self.locals.objective.variables)

    def update_objective_params(self, params=None, oid=None):
        if params is not None:
            self.locals.objective.set_variables(self.sess, params)
        if oid is not None:
            utils.plasma_prefetch([oid])
            feed_dict = {self.locals.plasma_var_oid: oid}
            self.sess.run(self.locals.plasma_read_vars, feed_dict)

    def compute_objective_gradients(self, t, grad_oid):
        self.feed_dict = self._generate_feed_dict(t, self.locals.large_rb_handle)

        if self.objective_vars_oid is not None:
            feed_dict = {self.locals.plasma_var_oid: self.objective_vars_oid}
            self.sess.run(self.locals.plasma_read_vars, feed_dict)
            self.objective_vars_oid = None

        self.feed_dict[self.locals.plasma_grad_oid] = grad_oid
        self.sess.run(self.locals.plasma_write_grads, self.feed_dict)
