from collections import namedtuple
import numpy as np
import tensorflow as tf
from baselines.a2c.utils import conv, fc, conv_to_fc
from baselines.common.tf_util import get_session
from baselines.common.tf_util import initialize

def cnn_small(layer_names=['c1', 'c2', 'fc1'], **conv_kwargs):
    def network_fn(X):
        h = tf.cast(X, tf.float32) / 255.

        activ = tf.nn.relu
        h = activ(conv(h, layer_names[0], nf=8, rf=8, stride=4, init_scale=np.sqrt(2), **conv_kwargs))
        h = activ(conv(h, layer_names[1], nf=16, rf=4, stride=2, init_scale=np.sqrt(2), **conv_kwargs))
        h = conv_to_fc(h)
        h = activ(fc(h, layer_names[2], nh=128, init_scale=np.sqrt(2)))
        return h
    return network_fn

SAVE_OBS_DIR = 'gdrive/MyDrive/182 Project/obs/'

class LevelSampler():
    def __init__(
        self, seeds, obs_space, action_space, num_actors=1, 
        strategy='random', replay_schedule='fixed', score_transform='power',
        temperature=1.0, eps=0.05,
        rho=0.2, nu=0.5, alpha=1.0, 
        staleness_coef=0, staleness_transform='power', staleness_temperature=1.0, save_obs_flag=False):
        self.obs_space = obs_space
        self.action_space = action_space
        self.strategy = strategy
        self.replay_schedule = replay_schedule
        self.score_transform = score_transform
        self.temperature = temperature
        self.eps = eps
        self.rho = rho
        self.nu = nu
        self.alpha = alpha
        self.staleness_coef = staleness_coef
        self.staleness_transform = staleness_transform
        self.staleness_temperature = staleness_temperature

        # Track seeds and scores as in np arrays backed by shared memory
        self._init_seed_index(seeds)

        self.unseen_seed_weights = np.array([1.]*len(seeds))
        self.seed_scores = np.array([0.]*len(seeds), dtype=np.float)
        self.partial_seed_scores = np.zeros((num_actors, len(seeds)), dtype=np.float)
        self.partial_seed_steps = np.zeros((num_actors, len(seeds)), dtype=np.int64)
        self.seed_staleness = np.array([0.]*len(seeds), dtype=np.float)

        self.next_seed_index = 0 # Only used for sequential strategy

        if self.strategy == 'rnd':
            self.sess = sess = get_session()
            pred_net = cnn_small(['c3','c4','fc2'])
            target_net = cnn_small(['c1','c2','fc1'])

            self.x = x = tf.placeholder(tf.float32, [None,64,64,3])
            y_model = pred_net(x)
            y = target_net(x)
            self.error = error = tf.math.reduce_mean(tf.square(y - y_model))
            train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'c3') + tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'c4') + tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'fc2')
            self.train_op = tf.train.GradientDescentOptimizer(0.01).minimize(error, var_list=train_vars)

            initialize()

            self.train_iters = 1
            self.random_sampling_iters = 150

        self.sampled_levels = []
        self.iter = 0

        self.value_avg = []
        self.value_var = []
        self.value_range = []
        self.entropy = []
        self.save_obs_flag = save_obs_flag

    def seed_range(self):
        return (int(min(self.seeds)), int(max(self.seeds)))

    def _init_seed_index(self, seeds):
        self.seeds = np.array(seeds, dtype=np.int64)
        self.seed2index = {seed: i for i, seed in enumerate(seeds)}

    def update_with_rollouts(self, rollouts):
        if self.strategy == 'random':
            return

        # Update with a RolloutStorage object
        if self.strategy == 'value_l1':
            score_function = self._average_value_l1
        elif self.strategy == 'policy_entropy':
            score_function = self._average_entropy
        elif self.strategy == 'rnd':
            score_function = self._rand_net_distillation
        else:
            raise ValueError(f'Unsupported strategy, {self.strategy}')

        self._update_with_rollouts(rollouts, score_function)

    def update_seed_score(self, actor_index, seed_idx, score, num_steps):
        score = self._partial_update_seed_score(actor_index, seed_idx, score, num_steps, done=True)

        self.unseen_seed_weights[seed_idx] = 0. # No longer unseen

        old_score = self.seed_scores[seed_idx]
        self.seed_scores[seed_idx] = (1 - self.alpha)*old_score + self.alpha*score

    def _partial_update_seed_score(self, actor_index, seed_idx, score, num_steps, done=False):
        partial_score = self.partial_seed_scores[actor_index][seed_idx]
        partial_num_steps = self.partial_seed_steps[actor_index][seed_idx]

        running_num_steps = partial_num_steps + num_steps
        merged_score = partial_score + (score - partial_score)*num_steps/float(running_num_steps)

        if done:
            self.partial_seed_scores[actor_index][seed_idx] = 0. # zero partial score, partial num_steps
            self.partial_seed_steps[actor_index][seed_idx] = 0
        else:
            self.partial_seed_scores[actor_index][seed_idx] = merged_score
            self.partial_seed_steps[actor_index][seed_idx] = running_num_steps

        return merged_score

    def _average_entropy(self, **kwargs):
        episode_logits = kwargs['episode_logits']
        num_actions = self.action_space.n
        max_entropy = -(1./num_actions)*np.log(1./num_actions)*num_actions

        return (-np.exp(episode_logits)*episode_logits).sum(-1).mean()/max_entropy

    def _average_value_l1(self, **kwargs):
        returns = kwargs['returns']
        value_preds = kwargs['value_preds']

        advantages = returns - value_preds

        return np.mean(np.abs(advantages))

    def _rand_net_distillation(self, **kwargs):
        if kwargs['obs'].shape[0] == 0:
          return 0
        
        loss_value = self.sess.run([self.error], feed_dict={self.x: kwargs['obs']})[0]

        return loss_value

    def rnd_update(self, obs):
        obs = obs.reshape((-1, 64, 64, 3))
        for i in range(self.train_iters):
            _, loss_value = self.sess.run([self.train_op, self.error], feed_dict={self.x: obs})

    def update_stats(self, value_preds, episode_logits):
        self.value_avg.append(np.mean(value_preds))
        self.value_var.append(np.var(value_preds))
        self.value_range.append(np.max(value_preds) - np.min(value_preds))
        num_actions = self.action_space.n
        max_entropy = -(1./num_actions)*np.log(1./num_actions)*num_actions
        self.entropy.append((-np.exp(episode_logits)*episode_logits).sum(-1).mean()/max_entropy)

    def update_staleness_coeff(self, episode_logits):
        num_actions = self.action_space.n
        max_entropy = -(1./num_actions)*np.log(1./num_actions)*num_actions
        new_staleness_coeff = (-np.exp(episode_logits)*episode_logits).sum(-1).mean()/max_entropy
        new_staleness_coeff = max(0, min(1, new_staleness_coeff))
        self.staleness_coef = new_staleness_coeff

    def save_obs(self, obs, returns, value_preds):
        advantages = returns - value_preds
        max_arg = np.argmax(np.abs(advantages))
        min_arg = np.argmin(np.abs(advantages))
        np.save(SAVE_OBS_DIR + 'iter_{}_advantage_{}.npy'.format(self.iter, np.round(np.abs(advantages[max_arg]), 3)), obs[max_arg])
        np.save(SAVE_OBS_DIR + 'iter_{}_advantage_{}.npy'.format(self.iter, np.round(np.abs(advantages[min_arg]), 3)), obs[min_arg])

    @property
    def requires_value_buffers(self):
        return self.strategy in ['value_l1']    

    @property
    def requires_obs_buffers(self):
        return self.strategy in ['rnd']    

    def _update_with_rollouts(self, rollouts, score_function):
        level_seeds = rollouts.level_seeds
        policy_logits = rollouts.action_log_dist
        done = ~(rollouts.masks > 0)
        total_steps, num_actors = policy_logits.shape[:2]
        num_decisions = len(policy_logits)

        if self.strategy == 'rnd':
            self.rnd_update(rollouts.obs)
        self.iter += 1

        if self.save_obs_flag:
            self.save_obs(rollouts.obs.reshape(-1, 64, 64, 3), rollouts.returns.reshape(-1), rollouts.value_preds.reshape(-1))

        for actor_index in range(num_actors):
            done_steps = np.array(done[:,actor_index].nonzero()).T[:total_steps,0]
            start_t = 0

            for t in done_steps:
                if not start_t < total_steps: break

                if t == 0: # if t is 0, then this done step caused a full update of previous seed last cycle
                    continue 

                seed_t = level_seeds[start_t,actor_index].item()
                seed_idx_t = self.seed2index[seed_t]

                score_function_kwargs = {}
                episode_logits = policy_logits[start_t:t,actor_index]
                score_function_kwargs['episode_logits'] = np.log(np.exp(episode_logits).T / np.sum(np.exp(episode_logits), axis=1)).T

                if self.requires_value_buffers:
                    score_function_kwargs['returns'] = rollouts.returns[start_t:t,actor_index]
                    score_function_kwargs['rewards'] = rollouts.rewards[start_t:t,actor_index]
                    score_function_kwargs['value_preds'] = rollouts.value_preds[start_t:t,actor_index]
                if self.requires_obs_buffers:
                    score_function_kwargs['obs'] = rollouts.obs[start_t:t,actor_index]

                score = score_function(**score_function_kwargs)

                num_steps = len(episode_logits)
                self.update_seed_score(actor_index, seed_idx_t, score, num_steps)

                start_t = t.item()

            if start_t < total_steps:
                seed_t = level_seeds[start_t,actor_index].item()
                seed_idx_t = self.seed2index[seed_t]

                score_function_kwargs = {}
                episode_logits = policy_logits[start_t:,actor_index]
                score_function_kwargs['episode_logits'] = np.log(np.exp(episode_logits).T / np.sum(np.exp(episode_logits), axis=1)).T

                if self.requires_value_buffers:
                    score_function_kwargs['returns'] = rollouts.returns[start_t:,actor_index]
                    score_function_kwargs['rewards'] = rollouts.rewards[start_t:,actor_index]
                    score_function_kwargs['value_preds'] = rollouts.value_preds[start_t:,actor_index]
                if self.requires_obs_buffers:
                    score_function_kwargs['obs'] = rollouts.obs[start_t:,actor_index]

                score = score_function(**score_function_kwargs)
                num_steps = len(episode_logits)
                self._partial_update_seed_score(actor_index, seed_idx_t, score, num_steps)

    def after_update(self):
        # Reset partial updates, since weights have changed, and thus logits are now stale
        for actor_index in range(self.partial_seed_scores.shape[0]):
            for seed_idx in range(self.partial_seed_scores.shape[1]):
                if self.partial_seed_scores[actor_index][seed_idx] != 0:
                    self.update_seed_score(actor_index, seed_idx, 0, 0)
        self.partial_seed_scores.fill(0)
        self.partial_seed_steps.fill(0)

    def _update_staleness(self, selected_idx):
        if self.staleness_coef > 0:
            self.seed_staleness = self.seed_staleness + 1
            self.seed_staleness[selected_idx] = 0

    def _sample_replay_level(self):
        sample_weights = self.sample_weights()

        if np.isclose(np.sum(sample_weights), 0):
            sample_weights = np.ones_like(sample_weights, dtype=np.float)/len(sample_weights)

        seed_idx = np.random.choice(range(len(self.seeds)), 1, p=sample_weights)[0]
        seed = self.seeds[seed_idx]

        self._update_staleness(seed_idx)

        self.sampled_levels.append(int(seed))

        if self.strategy == 'rnd' and self.iter < self.random_sampling_iters:
            seed_idx = np.random.choice(range((len(self.seeds))))
            seed = self.seeds[seed_idx]
            return int(seed)

        return int(seed)

    def _sample_unseen_level(self):
        sample_weights = self.unseen_seed_weights/self.unseen_seed_weights.sum()
        seed_idx = np.random.choice(range(len(self.seeds)), 1, p=sample_weights)[0]
        seed = self.seeds[seed_idx]

        self._update_staleness(seed_idx)

        return int(seed)

    def sample(self, strategy=None):
        if not strategy:
            strategy = self.strategy

        if strategy == 'random':
            seed_idx = np.random.choice(range((len(self.seeds))))
            seed = self.seeds[seed_idx]
            return int(seed)

        if strategy == 'sequential':
            seed_idx = self.next_seed_index
            self.next_seed_index = (self.next_seed_index + 1) % len(self.seeds)
            seed = self.seeds[seed_idx]
            return int(seed)

        num_unseen = (self.unseen_seed_weights > 0).sum()
        proportion_seen = (len(self.seeds) - num_unseen)/len(self.seeds)

        if self.replay_schedule == 'fixed':
            if proportion_seen >= self.rho: 
                # Sample replay level with fixed prob = 1 - nu OR if all levels seen
                if np.random.rand() > self.nu or not proportion_seen < 1.0:
                    return self._sample_replay_level()

            # Otherwise, sample a new level
            return self._sample_unseen_level()

        else: # Default to proportionate schedule
            if proportion_seen >= self.rho and np.random.rand() < proportion_seen:
                return self._sample_replay_level()
            else:
                return self._sample_unseen_level()

    def sample_weights(self):
        weights = self._score_transform(self.score_transform, self.temperature, self.seed_scores)
        weights = weights * (1-self.unseen_seed_weights) # zero out unseen levels

        z = np.sum(weights)
        if z > 0:
            weights /= z

        staleness_weights = 0
        if self.staleness_coef > 0:
            staleness_weights = self._score_transform(self.staleness_transform, self.staleness_temperature, self.seed_staleness)
            staleness_weights = staleness_weights * (1-self.unseen_seed_weights)
            z = np.sum(staleness_weights)
            if z > 0: 
                staleness_weights /= z

            weights = (1 - self.staleness_coef)*weights + self.staleness_coef*staleness_weights

        return weights

    def _score_transform(self, transform, temperature, scores):
        if transform == 'constant':
            weights = np.ones_like(scores)
        if transform == 'max':
            weights = np.zeros_like(scores)
            scores = scores[:]
            scores[self.unseen_seed_weights > 0] = -float('inf') # only argmax over seen levels
            argmax = np.random.choice(np.flatnonzero(np.isclose(scores, scores.max())))
            weights[argmax] = 1.
        elif transform == 'eps_greedy':
            weights = np.zeros_like(scores)
            weights[scores.argmax()] = 1. - self.eps
            weights += self.eps/len(self.seeds)
        elif transform == 'rank':
            temp = np.flip(scores.argsort())
            ranks = np.empty_like(temp)
            ranks[temp] = np.arange(len(temp)) + 1
            weights = 1/ranks ** (1./temperature)
        elif transform == 'power':
            eps = 0 if self.staleness_coef > 0 else 1e-3
            weights = (np.array(scores) + eps) ** (1./temperature)
        elif transform == 'softmax':
            weights = np.exp(np.array(scores)/temperature)

        return weights