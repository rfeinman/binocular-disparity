from tqdm import tqdm
import numpy as np
import tensorflow as tf

from .. import util

class GradientDescent(object):
    """
    A Markov Random Field model with methods to apply greedy gradient descent
    inference.

    :param height: [int]
    :param width: [int]
    :param num_beliefs: [int]
    :param session: [tf.Session] TensorFlow session to use
    :param alpha: [float] smoothness ratio
    :param beta: [float] truncation parameter for the binary potential function
    :param smooth_energies: [bool] whether we will be smoothing disparity
        energies. Default is False (smooth disparity values, not energies)
    """
    def __init__(
            self, height, width, num_beliefs, session, alpha=2., beta=0.2,
            smooth_energies=False
    ):
        assert isinstance(session, tf.Session)
        if smooth_energies:
            shape = (height, width, num_beliefs)
        else:
            shape = (height, width)
        self.smooth_energies = smooth_energies
        self.height = height
        self.width = width
        self.num_beliefs = num_beliefs
        self.sess = session
        self.alpha = alpha
        self.beta = beta

        ## Build the graph ##

        # observations 'y'
        y = tf.placeholder(shape=shape, dtype=tf.float32)
        # perturbation
        delta = tf.Variable(np.zeros(shape), dtype=tf.float32)
        # latent variables 'x'
        x = y + delta
        x_flat = tf.reshape(x, (height*width, -1))
        pairs = util.get_neighboring_pairs(height, width)
        # data-fitting potential
        fit = tf.reduce_sum(tf.square(delta))

        # smoothness potential
        squared_diff = tf.square(
            tf.gather(x_flat, pairs[:,0]) - tf.gather(x_flat, pairs[:,1])
        )
        smoothness = tf.reduce_sum(squared_diff / (squared_diff + beta))

        # total loss
        loss = fit + alpha * smoothness

        self.y = y
        self.x = x
        self.delta = delta
        self.loss = loss


    def decode_MAP(self, observations, lr=0.01, iterations=100):
        """
        Perform CRF smoothing via greedy gradient-descent method; decode the
        latent variables of the CRF model

        :param observations: [(H,W) or (H,W,n) ndarray] the CRF observations
        :param lr: [float] learning rate
        :param iterations: [int] number of optimization steps
        :return belief [(H,W) or (H,W,n) ndarray] the decoded latent variables
        """
        # record
        assert len(observations.shape) in [2,3]
        assert observations.shape[:2] == (self.height, self.width)

        if self.smooth_energies:
            # normalize the observations
            observations = observations - observations.mean()
            observations = observations / observations.std()

        # optimizer
        optimizer = tf.train.RMSPropOptimizer(lr)
        train_op = optimizer.minimize(self.loss, var_list=[self.delta])

        # perform gradient descent
        loss_vals = []
        with self.sess.as_default() as sess:
            init_vars = tf.variables_initializer(
                [self.delta] + optimizer.variables()
            )
            sess.run(init_vars)
            for _ in tqdm(range(iterations)):
                _, loss_val = sess.run(
                    [train_op, self.loss],
                    feed_dict={self.y:observations}
                )
                loss_vals.append(loss_val)
            # obtain final smoothed energies
            belief = sess.run(self.x, feed_dict={self.y:observations})

        if self.smooth_energies:
            belief = np.argmin(belief, axis=2)

        return belief