import tensorflow as tf
from keras import initializers, layers

class ActorNetwork:
    """
    Input to the network is the state, output is the action
    under a deterministic policy.
    """

    def __init__(self, sess, h1, h2, s_dim, a_dim, action_low, action_high, lr, tau, batch_size):
        self.sess = sess
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.action_low = action_low
        self.action_high = action_high
        self.action_range = action_high - action_low # for scaling sigmoid output
        self.h1 = h1 # number of nodes in hidden layer 1
        self.h2 = h2 # number of nodes in hidden layer 2
        self.lr = lr # learning rate
        self.tau = tau
        self.batch_size = batch_size
        
        # local actor
        self.state, self.action, self.nn_params = \
        self.create_actor_network(model_name="local_actor")
        
        # target actor
        self.target_state, self.target_action, self.target_nn_params = \
        self.create_actor_network(model_name="target_actor")
        
        # periodically update target actor
        with tf.variable_scope("update_target_actor"):
            self.update_target_nn_params = [self.target_nn_params[i].assign(
                tf.multiply(self.nn_params[i], self.tau) 
                + tf.multiply(self.target_nn_params[i], 1. - self.tau)) 
                                                 for i in range(len(self.target_nn_params))]
        
        # a placeholder for injecting external data, i.e., from critic
        self.dQ_da = tf.placeholder(tf.float32, [None, self.a_dim], name="dQ_da")

        with tf.variable_scope("optimize_actor"):
            # da_dtheta * (-dQ_da) 
            self.params_grad_raw = tf.gradients(self.action, self.nn_params, -self.dQ_da,
                                               name="gradients_params_raw")
            
            # raw gradients are summed over all samples in a minibatch, need to normalize
            with tf.variable_scope("normalize_gradients_batch"):
                self.params_grad = list(map(lambda x: tf.div(x, self.batch_size), self.params_grad_raw))
            
            # Actor optimization
            grads_vars = zip(self.params_grad, self.nn_params)
            self.optimize = tf.train.AdamOptimizer(self.lr).apply_gradients(grads_vars)

    def create_actor_network(self, model_name):
        """create NN model useing keras"""
        state = layers.Input(shape=(self.s_dim,), name="state") # input
        with tf.variable_scope(model_name):
            net = layers.Dense(self.h1, name="l1-dense-{}".format(self.h1))(state) # pre-activations
            net = layers.Activation('relu', name="relu1")(net) # activated
            net = layers.Dense(self.h2, name="l2-dense-{}".format(self.h2))(net)
            net = layers.Activation('relu', name="relu2")(net)
            w_init = initializers.RandomUniform(minval=-3e-3, maxval=3e-3)
            net = layers.Dense(self.a_dim, kernel_initializer=w_init, name="l3-dense-{}".format(self.a_dim))(net)
            raw_action = layers.Activation('sigmoid', name="sigmoid")(net) # raw action: [0, 1]
            action = layers.Lambda(lambda x: x * self.action_range + self.action_low, 
                                   name="scaled_action")(raw_action) # output
            nn_params = tf.trainable_variables(scope=model_name) # NN params
        return state, action, nn_params
        
    def train(self, state, a_gradient):
        self.sess.run(self.optimize, feed_dict={
            self.state: state, 
            self.dQ_da: a_gradient
        })

    def predict(self, state):
        return self.sess.run(self.action, feed_dict={
            self.state: state
        })

    def predict_target(self, state):
        return self.sess.run(self.target_action, feed_dict={
            self.target_state: state
        })

    def update_target_network(self):
        self.sess.run(self.update_target_nn_params)
        
    
class CriticNetwork:
    """
    Input to the network is the state and action, output is Q(s,a).
    The action must be obtained from the output of the Actor network.
    """
    def __init__(self, sess, h1, h2, s_dim, a_dim, lr, tau, gamma):
        
        self.sess = sess
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.h1 = h1 # number of nodes in hidden layer 1
        self.h2 = h2 # number of nodes in hidden layer 2
        self.lr = lr
        self.tau = tau
        self.gamma = gamma
        
        # local critic model
        self.state, self.action, self.Qhat, self.nn_params = \
        self.create_critic_network(model_name="local_critic")
        
        # target critic model
        self.target_state, self.target_action, self.target_Qhat, self.target_nn_params = \
        self.create_critic_network(model_name="target_critic")
        
        # periodically update target critic
        with tf.variable_scope("update_target_critic"):
            self.update_target_nn_params = [self.target_nn_params[i].assign(
                tf.multiply(self.nn_params[i], self.tau) 
                + tf.multiply(self.target_nn_params[i], 1. - self.tau)) 
                                                 for i in range(len(self.target_nn_params))]

        # Critic target output
        # a placeholder for injecting external data, i.e., from TD target
        self.Q_from_TD_target = tf.placeholder(tf.float32, [None, 1], name="Q_true")

        with tf.variable_scope("optimize_critic"):
            # Define loss and optimization Op
            self.loss = tf.losses.mean_squared_error(self.Q_from_TD_target, self.Qhat)
            self.optimize = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

        # dQ_da: gradient of the critic output w.r.t. action
        self.dQ_da = tf.gradients(self.Qhat, self.action, name="gradients_dQ_da")

    def create_critic_network(self, model_name):
        """create NN model useing keras"""
        state = layers.Input(shape=(self.s_dim,), name="state")
        action = layers.Input(shape=(self.a_dim,), name="action")
        with tf.variable_scope(model_name):
            net = layers.Dense(self.h1, name="l1-dense-{}".format(self.h1))(state)
            net = layers.Activation('relu', name="relu1")(net)
            net = layers.Concatenate()([net, action]) # combined state and action
            net = layers.Dense(self.h2, name="l2-dense-{}".format(self.h2))(net)
            net = layers.Activation('relu', name="relu2")(net)
            w_init = initializers.RandomUniform(minval=-3e-3, maxval=3e-3) # 3e-3 better than 3e-4 in ddpg paper
            Qhat = layers.Dense(1, kernel_initializer=w_init, name="l3-dense-1")(net) # output
            nn_params = tf.trainable_variables(scope=model_name) # NN params
        return state, action, Qhat, nn_params
        
    def train(self, state, action, Q_from_TD_target):
        return self.sess.run([self.Qhat, self.optimize], feed_dict={
            self.state: state,
            self.action: action,
            self.Q_from_TD_target: Q_from_TD_target
        })

    def predict(self, state, action):
        return self.sess.run(self.Qhat, feed_dict={
            self.state: state, 
            self.action: action
        })

    def predict_target(self, state, action):
        return self.sess.run(self.target_Qhat, feed_dict={
            self.target_state: state, 
            self.target_action: action
        })

    def action_gradients(self, state, action):
        return self.sess.run(self.dQ_da, feed_dict={
            self.state: state,
            self.action: action
        })

    def update_target_network(self):
        self.sess.run(self.update_target_nn_params)