from tqdm import tqdm
import numpy as np
import tensorflow as tf

from . import util

# note: coordinates are assumed to be such that "y" increases "downwards",
# and "x" increases "rightwards".
# arrays will generally be indexed as array[y][x].

# directions are as indexed in arrays,
# defined as pseudo-constants here for convenience.
# base comes last as it's not used in the working array.
RIGHT = 0
UP = 1
LEFT = 2
DOWN = 3
BASE = 4
DIRECTIONS = {"right":RIGHT, "up":UP, "left":LEFT, "down":DOWN, "base":BASE}



class LoopyBP(object):
    def __init__(self, height, width, num_beliefs):
        """Initialize the MRF with given height, width and number of beliefs.

        init_base_belief() must be called manually before calling
        pass_messages().
        """

        # basic dimensions
        self.width = width
        self.height = height
        self.num_beliefs = num_beliefs



class MaxProduct(LoopyBP):
    def __init__(self, height, width, num_beliefs, damping=0.5, tol=1e-5):
        super(MaxProduct, self).__init__(height, width, num_beliefs)
        self.damping = damping
        self.tol = tol

        # store unary and binary potential functions as arrays
        unaries = np.zeros((num_beliefs, num_beliefs), dtype=np.float32)
        binaries = np.zeros((num_beliefs, num_beliefs), dtype=np.float32)
        for a in range(num_beliefs):
            for b in range(num_beliefs):
                unaries[a, b] = -unary_potential(a, b)
                binaries[a, b] = -binary_potential(a, b)
        self.unaries = unaries
        self.binaries = binaries

        # get list of edges
        self.edges = get_neighboring_pairs(height, width)

    def decode_MAP(self, observations, iterations=30):
        assert len(observations.shape) == 2
        global unary_potentials, pairwise_potentials, edges, damping, \
            messages, all_incoming

        # one-hot-encode the observations
        features = np.zeros(
            (self.height*self.width, self.num_beliefs),
            dtype=np.float32
        )
        features[np.arange(self.height*self.width), observations.flatten()] = 1

        # store edges
        edges = self.edges

        # get potentials
        unary_potentials = np.dot(features, self.unaries.T)
        pairwise_potentials = np.repeat(
            self.binaries[np.newaxis,:,:], edges.shape[0], axis=0
        )

        # iterative max product
        damping = self.damping
        n_edges = len(edges)
        n_vertices, n_states = unary_potentials.shape
        messages = np.zeros((n_edges, 2, n_states))
        all_incoming = np.zeros((n_vertices, n_states))
        for i in range(iterations):
            diff = 0
            #results = [maxprod_iteration(e) for e in range(n_edges)]
            results = util.parallel(maxprod_iteration, range(n_edges))
            for e, (edge, result) in enumerate(zip(edges, results)):
                a, b = edge
                message_0, message_1, update_a, update_b, d = result
                messages[e, 0] = message_0
                messages[e, 1] = message_1
                all_incoming[a] += update_a
                all_incoming[b] += update_b
                diff += d
            if diff < self.tol:
                break

        belief = np.argmax(all_incoming + unary_potentials, axis=1)
        belief = belief.reshape(self.height, self.width)

        return belief

def unary_potential(x, y):
    return np.square(x - y)

def binary_potential(a, b, alpha=60., beta=1.5):
    d = np.square(a - b)
    smooth = alpha * d / (d + beta)

    return smooth

def maxprod_iteration(e):
    a, b = edges[e]
    pairwise = pairwise_potentials[e]
    # update message from edge[0] to edge[1]
    update = all_incoming[a] + pairwise.T + unary_potentials[a] - messages[e,1]
    old_message = messages[e,0].copy()
    new_message = np.max(update, axis=1)
    new_message = new_message - np.max(new_message)
    new_message = damping * old_message + (1 - damping) * new_message
    message_0 = new_message
    update_b = new_message - old_message

    # update message from edge[1] to edge[0]
    update = all_incoming[b] + pairwise + unary_potentials[b] - messages[e,0]
    old_message = messages[e, 1].copy()
    new_message = np.max(update, axis=1)
    new_message = new_message - np.max(messages[e, 1])
    new_message = damping * old_message + (1 - damping) * new_message
    message_1 = new_message
    update_a = new_message - old_message

    diff = np.abs(update_a).sum() + np.abs(update_b).sum()

    return (message_0, message_1, update_a, update_b, diff)



class SumProduct(LoopyBP):
    """
    A Markov Random Field with methods to apply Loopy Belief Propagation.
    """

    def __init__(self, height, width, num_beliefs):
        """Initialize the MRF with given height, width and number of beliefs.

        init_base_belief() must be called manually before calling
        pass_messages().
        """
        super(SumProduct, self).__init__(height, width, num_beliefs)

    def decode_MAP(self, observations, iterations=20):
        assert len(observations.shape) == 3
        # convert energies into probabilities
        probs = util.energies_to_probs(observations)

        # initialize the model
        self.init_MRF()
        self.init_smoothness()
        self.init_base_belief(probs)

        # perform loopy BP iterations
        for _ in tqdm(range(iterations)):
            self.pass_messages()

        # compute the final decoded MAP
        belief = self.calc_belief()

        return belief

    def init_MRF(self):
        # main data array
        self.data = np.ones(
            shape=(self.height, self.width, 5, self.num_beliefs),
            dtype=np.float32)
        self.data /= self.num_beliefs

        # working array
        self._working = np.ones(
            shape=(self.height,self.width,4,self.num_beliefs,self.num_beliefs),
            dtype=np.float32
        )
        self._working /= self.num_beliefs

        # belief storage arrays
        self._beliefprod = np.ndarray(
            shape=(self.height, self.width, self.num_beliefs),
            dtype=np.float32
        )
        self._belief = np.ndarray(
            shape=(self.height, self.width),
            dtype=np.int
        )

        # normalization temporary storage array
        self._sumstorage = np.ndarray(
            shape=(self.height, self.width),
            dtype=np.float32
        )

        # note that filling the data arrays with ones is important,
        # as messages from outside the array will not be modified.
        # that is, the messages from the outer edge pointing inwards
        # are never updated from their initial values here.
        # initializing to 1 means they have no effect when multiplying,
        # letting us avoid special-casing edge behaviour.

        # all data arrays are also normalized across possible beliefs,
        # for convenience and stability

    def init_base_belief(self, base_beliefs):
        """Initialize the base belief channel.

        Input should have the same height, width, and number of beliefs
        as the underlying MRF.

        Values should be the relative likelihood of each belief possibility.
        """

        # perform basic sanity checks and fail noisily
        if len(base_beliefs) != len(self.data) \
        or len(base_beliefs[0]) != len(self.data[0]):
            raise Exception("belief dimensions (%s,%s) don't match MRF "
                            "dimensions (%s,%s)" %
                            (len(base_beliefs), len(base_beliefs[0]),
                             len(self.data), len(self.data[0])))
        if len(base_beliefs[0][0]) != len(self.data[0][0][0]):
            raise Exception("number of belief possibilities must match MRF")

        # now normalize the data while copying it in
        self.data[:,:,DIRECTIONS["base"],:] = base_beliefs * (
                1 / np.sum(base_beliefs,axis=2,keepdims=True) )

    def init_smoothness(self):
        """Initialize the smoothness array.
        
        It should have dimension num_beliefs * num_beliefs,
        and ideally be symmetric.
        """
        smoothness = np.ndarray(
            shape=(self.num_beliefs, self.num_beliefs), dtype=np.float32
        )
        for a in range(self.num_beliefs):
            for b in range(self.num_beliefs):
                smoothness[a][b] = howsmooth(a, b)

        # normalize and store the smoothness array
        self.smoothness = smoothness * (1 / np.sum(smoothness,axis=1,keepdims=True))

    def pass_messages(self, direction=None):
        """Pass messages in the specified direction.
        
        If no direction is specified, messages are passed in all directions.
        Right, then up, then left, then down.
        """

        # if no direction specified, pass in each direction
        if direction is None:
            self.pass_messages(RIGHT)
            self.pass_messages(UP)
            self.pass_messages(LEFT)
            self.pass_messages(DOWN)
            return

        # otherwise interpret the given direction.
        # As we will be passing messages across pairs in some direction,
        # the working area is smaller by one pixel in that axis,
        # and shifted by one pixel between from and to.
        # If we represent the from and to areas with slices,
        # indices and sizes will match up perfectly.
        # The direction of message passing and its opposite are also stored,
        # as later we discount messages from the pixel we're passing to.
        if direction == RIGHT:
            working_slice = self._working[:,:-1,RIGHT]
            from_slice = self.data[:,:-1]
            to_slice = self.data[:,1:]
            from_dir = LEFT
            to_dir = RIGHT
            storage = self._sumstorage[:,1:]
        elif direction == UP:
            working_slice = self._working[1:,:,UP]
            from_slice = self.data[1:,:]
            to_slice = self.data[:-1,:]
            from_dir = DOWN
            to_dir = UP
            storage = self._sumstorage[:-1,:]
        elif direction == LEFT:
            working_slice = self._working[:,1:,LEFT]
            from_slice = self.data[:,1:]
            to_slice = self.data[:,:-1]
            from_dir = RIGHT
            to_dir = LEFT
            storage = self._sumstorage[:,:-1]
        elif direction == DOWN:
            working_slice = self._working[:-1,:,DOWN]
            from_slice = self.data[:-1,:]
            to_slice = self.data[1:,:]
            from_dir = UP
            to_dir = DOWN
            storage = self._sumstorage[1:,:]
        elif direction == BASE:
            raise Exception("can't pass messages to base belief channel")
        else:
            raise Exception("invalid direction index: %s" % direction)

        # for now the algorithm used is the "sum-product" algorithm.
        # it goes as follows:
        #
        # for each pixel (x,y coord)
        # for each direction (left/right/up/down)
        # we're passing a message.
        # The message is a vector assigning weights to each belief possibility.
        # To calculate each element of the message,
        # multiply the base probability of this element given the data
        # by the sum over all belief possibilities of
        # the product of
        #       the similarity of this possibility to the message element
        # and   the product of the beliefs regarding this element
        #       in all incoming messages
        #       OTHER than the one from the pixel we're sending to,
        # then normalize and send the message.

        # here goes the implementation.

        # first initialize our working area with the smoothness function.
        # The order of multiplication doesn't actually matter here,
        # but this is as good a place to start as any.
        # We slice the slice so that numpy copies into the existing memory,
        # in stead of just making 'working_slice' a reference to smoothness.
        # This will copy the smoothness array into the working area
        # for every pixel in our working slice.
        working_slice[:] = self.smoothness

        # multiply by the base data.
        # This weights the output elements according to the base belief.
        # we want working_slice[i] *= base_data[i] for i in num_beliefs,
        # so we need to mess with the axes a little,
        # but we can do that by adding an axis to the base belief data,
        # which should be efficiently done by numpy.
        # (no new memory allocations here, AFAIK.)
        working_slice[:] *= from_slice[:,:,BASE,:,np.newaxis]

        # the three messages not from the direction we're sending to
        # will each be multiplied into our extra working dimension,
        # which will be summed in the next step.
        # Because we initialized with the smoothness array,
        # these are automatically being weighted by our smoothness function.
        # We need to specify the axis to broadcast here as well,
        # but it's transposed relative to the base data.
        for d in (RIGHT, UP, LEFT, DOWN):
            # don't include the message from the pixel we're sending to
            if d == to_dir: continue
            # but do include the other three
            working_slice[:] *= from_slice[:,:,d,np.newaxis,:]

        # sum the extra working axis to get the message we want.
        # Using numpy's sum function lets us sum directly into the output.
        # Axes 0 and 1 are the x and y coordinates of each pixel,
        # axis 2 is the desired output axis,
        # so we sum across axis 3.
        # (note that axes 2 and 3 could have been swapped,
        # this just needs to be consistent with the operations above.)
        np.sum(working_slice, axis=3, out=to_slice[:,:,from_dir])

        # now normalize the message.
        # This does not change the belief,
        # but if we do not normalize, values will decrease each iteration
        # until floating point limits are hit.
        np.sum(to_slice[:,:,from_dir], axis=2, out=storage)
        np.reciprocal(storage, out=storage)
        to_slice[:,:,from_dir] *= storage[:,:,np.newaxis]

    def calc_belief(self):
        """Calculate the index of the most likely belief at each pixel.
        """
        # reuses storage, if you want to keep it, copy it
        self.data.prod(axis=2, out=self._beliefprod)
        self._beliefprod.argmax(axis=2, out=self._belief)
        return self._belief

    def __repr__(self):
        """Represent the MRF by it's data for now.
        """
        return repr(self.data)

def howsmooth(a,b,threshold=0.1):
    s = np.exp(-np.square(a-b))
    s = max(s,threshold)

    return s



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
    """
    def __init__(self, height, width, num_beliefs, session, alpha=2., beta=0.2):
        shape = (height, width, num_beliefs)
        assert isinstance(session, tf.Session)
        self.height = height
        self.width = width
        self.num_beliefs = num_beliefs
        self.sess = session
        self.alpha = alpha
        self.beta = beta
        self.pairs = get_neighboring_pairs(height, width)

        ## Build the graph ##

        # observations 'y'
        y = tf.placeholder(shape=shape, dtype=tf.float32)
        # perturbation
        delta = tf.Variable(np.zeros(shape), dtype=tf.float32)
        # latent variables 'x'
        x = y + delta
        x_flat = tf.reshape(x, (height*width, num_beliefs))
        pairs = get_neighboring_pairs(height, width)
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
        Perform CRF smoothing via greedy gradient-descent method; decode the latent
        variables of the CRF model

        :param observations: [(H,W) or (H,W,n) ndarray] the CRF observations
        :param lr: [float] learning rate
        :param iterations: [int] number of optimization steps
        :return x_smooth: [(H,W) or (H,W,n) ndarray] the decoded latent variables
        :return losses: [list of float] losses for each iteration
        """
        # record
        assert len(observations.shape) in [2,3]
        assert observations.shape[:2] == (self.height, self.width)

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
            energies_smooth = sess.run(self.x, feed_dict={self.y:observations})

        belief = np.argmin(energies_smooth, axis=2)

        return belief


def get_neighboring_pairs(height, width):
    x = np.array(range(height*width))
    x = x.reshape(height, width)
    pairs = []
    for i in range(height):
        pairs += zip(x[i], x[i,1:])
    for j in range(width):
        pairs += zip(x[:,j], x[1:,j])
    pairs = np.asarray(pairs, dtype=np.int32)

    return pairs