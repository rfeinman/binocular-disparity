import numpy as np

from .. import util



class MaxProduct(object):
    """
    A Markov Random Field with methods to apply the sum-product Loopy Belief
    Propagation inference algorithm.
    """
    def __init__(self, height, width, num_beliefs, damping=0.5, tol=1e-5):
        # basic params
        self.width = width
        self.height = height
        self.num_beliefs = num_beliefs
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
        self.edges = util.get_neighboring_pairs(height, width)

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