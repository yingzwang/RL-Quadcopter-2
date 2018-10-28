import time
import itertools
from collections import deque
import random
import numpy as np
import tensorflow as tf

from agents.actor_critic import ActorNetwork, CriticNetwork

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size=int(1e5), random_seed=1234):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size: maximum size of buffer
            The right side of the deque contains the most recent experiences. 
        """
        self.buffer_size = buffer_size
        self.buffer = deque(maxlen=buffer_size)
        random.seed(random_seed)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.buffer)
    
    def add(self, s, a, r, done, s2):
        """Add a new experience to buffer.
        Params
        ======
        s: one state sample, numpy array
        a: one action sample, numpy array
        r: one reward sample, scalar
        done: True/False, scalar
        s2: one state sample, numpy array
        """
        # flatten state and action, s.t. len(s) returns s_dim, len(a) returns a_dim
        e = (s.ravel(), a.ravel(), r, done, s2.ravel()) 
        self.buffer.append(e)
        
    def sample_batch(self, batch_size):
        """Randomly sample a batch of experiences from buffer."""
        # ensure the buffer is large enough for sampleling 
        assert (len(self.buffer) >= batch_size)
        
        batch = random.sample(self.buffer, batch_size)
        s, a, r, done, s2 = batch[0] # get one experience
        s_dim = len(s)
        a_dim = len(a)
        Ns = batch_size
        
        # Convert experience tuples to separate arrays for each element (states, actions, rewards, etc.)
        states, actions, rewards, dones, next_states = zip(*batch)
        states = np.asarray(states).reshape(Ns, s_dim)
        actions = np.asarray(actions, dtype=np.float32).reshape(Ns, a_dim)
        rewards = np.asarray(rewards, dtype=np.float32).reshape(Ns, 1)
        next_states = np.asarray(next_states).reshape(Ns, s_dim)
        dones = np.asarray(dones, dtype=np.uint8).reshape(Ns, 1)
        return states, actions, rewards, dones, next_states

    
class OUNoise:
    """ 
    # generate ornstein-uhlenbeck noise 
    # based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
    """
    def __init__(self, ndim, mu=.0, sigma=0.3, theta=.15, dt=1e-2, x0=None):
        self.mu = mu
        self.sigma = sigma
        self.theta = theta
        self.dt = dt
        self.x0 = x0
        self.ndim = ndim
        self.reset()

    def reset(self):
        if self.x0 is not None:
            self.xn = self.x0
        else:
            self.xn = np.ones(self.ndim) * self.mu

    def sample(self):
        dw = self.sigma * np.sqrt(self.dt) * np.random.randn(self.ndim)
        dx= self.theta * (self.mu - self.xn) * self.dt + dw
        self.xn += dx
        return self.xn
    
    def __repr__(self):
        return 'OUNoise(mu={}, sigma={}, theta={}, dt={})'.format(self.mu, self.sigma, self.theta, self.dt)

    
def build_summaries(actor, critic):
    """
    tensorboard summary for monitoring training process
    """
    # NN weights per episode
    for var in itertools.chain(actor.nn_params, critic.nn_params):
        vname = var.name.replace("kernel:0", "W").replace("bias:0", "b")
        tf.summary.histogram(vname, var)

    # performance per episode
    ph_reward = tf.placeholder(tf.float32) 
    ph_Qmax = tf.placeholder(tf.float32)
    tf.summary.scalar("Reward", ph_reward)
    tf.summary.scalar("Qmax_Value", ph_Qmax)

    # merged summary op
    summary_op = tf.summary.merge_all()
    
    return summary_op, ph_reward, ph_Qmax

class ddpg_agent:
    """DDPG agent"""

    def __init__(self, sess, task, args):
        # state and action params obtained from task
        s_dim = task.state_size
        a_dim = task.action_size
        action_low = task.action_low    # known action range
        action_high = task.action_high  # known action range

        # initialize actor
        self.actor = ActorNetwork(sess, int(args['actor_h1']), int(args['actor_h2']), s_dim, a_dim, action_low, 
                             action_high, float(args['actor_lr']), float(args['tau']), int(args['minibatch_size']))

        # initialize critic
        self.critic = CriticNetwork(sess, int(args['critic_h1']), int(args['critic_h2']), s_dim, a_dim,
                               float(args['critic_lr']), float(args['tau']), float(args['gamma']))

        # initialize action noise
        self.actor_noise = OUNoise(a_dim, float(args['ou_mu']), float(args['ou_sigma']), 
                              float(args['ou_theta']), float(args['ou_dt']))

        # initialize replay memory
        self.replay_buffer = ReplayBuffer(int(args['buffer_size']), int(args['random_seed']))
        self.minibatch_size = int(args['minibatch_size'])

        # initialize summary (for visualization in tensorboard)
        self.summary_op, self.ph_reward, self.ph_Qmax = build_summaries(self.actor, self.critic)
        subdir = time.strftime("%Y%m%d-%H%M%S", time.localtime()) # a sub folder, e.g., yyyymmdd-HHMMSS
        logdir = args['summary_dir'] + '/' + subdir
        self.writer = tf.summary.FileWriter(logdir, sess.graph) # must be done after graph is constructed
        
        # initialize variables existed in the graph
        sess.run(tf.global_variables_initializer()) # must be done after graph is constructed


    def train(self, sess, task, args):
        num_ep = args['max_episodes']
        num_ep_explore = args['max_explore_episodes']
        max_t = args['max_episode_len']

        # saved results for plotting
        labels = ['time_all', 'pos_all', 'v_all', 'action_all', 'a_clean_all', 'a_noise_all', 'reward_all']
        results = {}
        for x in labels:
            if x == 'pos_all' or x == 'v_all':
                results[x] = np.full((num_ep, max_t, 3), np.nan)
            elif x == 'action_all' or x == 'a_clean_all' or x == 'a_noise_all':
                results[x] = np.full((num_ep, max_t, self.actor.a_dim), np.nan)
            else:
                results[x] = np.full((num_ep, max_t), np.nan)

        # loop over episodes
        for ep in range(num_ep):

            s = task.reset()
            self.actor_noise.reset()

            ep_reward = 0
            ep_ave_qmax = 0
            t_step = 0
            p = min(ep/num_ep_explore, 1) # p increased from 0 to 1 in "num_ep_explore" episodes
            while t_step <= max_t:

                # predict action using the current policy
                a_clean = self.actor.predict(np.reshape(s, (1, self.actor.s_dim)))[0] # shape (a_dim,)

                # exploration noise
                a_noise = self.actor_noise.sample()  # shape (a_dim,)

                a_noise *= (1 - p) # noise weight (1 - p) decreased from 1 to 0 in "num_ep_eplore" episodes
                
                a = a_clean + a_noise # predicted action with added noise
                
                """
                # epsilon greedy
                epsilon = 0.0 # probability of exploration
                if epsilon > np.random.rand():
                    # explore using random action
                    a = np.random.rand(self.actor.a_dim) * 900 
                else:
                    # greedy action
                    a = a_clean + a_noise # predicted action with added noise
                """
                
                """
                # Clipping action is not needed. Exploding action will be punished by rewards and be learned.
                a = np.clip(a, self.actor.action_low, self.actor.action_high)
                """
                
                # interact with env, and add the experience to memory
                s2, r, done = task.step(a)
                self.replay_buffer.add(s, a, r, done, s2)

                # learn from a batch
                if len(self.replay_buffer) > 3 * self.minibatch_size:
                    # Sample a batch
                    s_batch, a_batch, r_batch, done_batch, s2_batch = self.replay_buffer.sample_batch(self.minibatch_size)

                    # a fixed target for critic
                    target_q = self.critic.predict_target(s2_batch, self.actor.predict_target(s2_batch)) 
                    y_batch = r_batch + self.critic.gamma * target_q * (1 - done_batch) # shape (minibatch_size, 1)

                    # train critic
                    Qhat_train, _ = self.critic.train(s_batch, a_batch, y_batch)
                    ep_ave_qmax += np.max(Qhat_train)

                    # train actor
                    a_outs = self.actor.predict(s_batch)
                    grads = self.critic.action_gradients(s_batch, a_outs) # grads is a list of one numpy array
                    self.actor.train(s_batch, grads[0])# grads[0] shape (minibatch_size, a_dim) 

                    # Update target NN
                    self.actor.update_target_network()
                    self.critic.update_target_network()

                # save per time step results for plotting
                to_write = [task.sim.time] + [task.sim.pose[:3]] + [task.sim.v] + [a] + [a_clean] + [a_noise] + [r]
                for ii in range(len(labels)):
                    results[labels[ii]][ep, t_step] = to_write[ii]

                # next time step
                t_step += 1
                s = s2 
                ep_reward += r 

                # episode ends
                if done:
                    break

            if ep % 10 == 0:
                print("episode: {}, steps: {}, reward: {:.2f}".format(ep, t_step, ep_reward))

            summary_str = sess.run(self.summary_op, feed_dict={self.ph_reward: ep_reward, self.ph_Qmax: ep_ave_qmax/t_step})
            self.writer.add_summary(summary_str, ep)
            self.writer.flush()

        return results, labels
    
    def test(self, task, max_t):
        """
        test a trained DDPG agent for one episode
        """
        labels = ['time', 'x', 'y', 'z', 'phi', 'theta', 'psi', 'x_velocity', 'y_velocity', 'z_velocity', 
                  'phi_velocity', 'theta_velocity', 'psi_velocity', 
                  'rotor_speed1', 'rotor_speed2', 'rotor_speed3', 'rotor_speed4']
        results = {x : [] for x in labels}
        done = False
        t = 0
        state = task.reset()
        while (t < max_t) and (not done):
            action = self.actor.predict_target(np.reshape(state, (1, self.actor.s_dim)))[0] # shape (a_dim,)
            state, reward, done = task.step(action)
            t += 1
            to_write = [task.sim.time] + list(task.sim.pose) + list(task.sim.v) + list(task.sim.angular_v) + list(action)
            for ii in range(len(labels)):
                results[labels[ii]].append(to_write[ii])
            # for plot trajectory
            poses = np.vstack((results['x'], results['y'], results['z'])).T # (max_t, 3)
            pos_all = np.expand_dims(poses, 0) # (1, maxt, 3)
            
        return results, pos_all
        

