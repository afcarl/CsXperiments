import numpy as np

from csxnet.brainforge.layers import _FCLayer
from csxnet.util import white
from csxnet.brainforge.activations import Sigmoid, Tanh

sigmoid, tanh = Sigmoid(), Tanh()


class LSTM(_FCLayer):
    def __init__(self, brain, neurons, inputs, position, activation=Tanh):
        _FCLayer.__init__(self, brain, neurons, position, activation)

        Z = neurons + inputs

        # Problem (?):
        # These are all input weights, we are not using the previous output!!
        # Idk whether this is a problem tho

        self.weights = white(Z, neurons*4)
        self.biases = white(1, neurons*4)

        self.gate_W_gradients = np.zeros_like(self.weights)
        self.gate_b_gradients = np.zeros_like(self.biases)

        self.states = []
        self.outputs = []
        self.caches = []

        self.time = 0
        self.fanin = inputs

    def feedforward(self, stimuli: np.ndarray):
        self.outputs = []
        self.states = [np.zeros((self.brain.m, self.neurons))]
        self.caches = []
        self.time = stimuli.shape[1]
        self.inputs = stimuli
        # this step might be neccesary if the first datadim is not time, but the batch index
        # stimuli = np.transpose(stimuli, (1, 0, 2))

        n = self.neurons

        def timestep(t):
            # Sizes:        (bsize, embed)  (1, neurons)
            X = np.column_stack((stimuli[t], self.outputs[-1]))
            gates = X.dot(self.weights) + self.biases
            gates[:, :n * 3] = sigmoid(gates[:, n * 3])
            gates[:, 3 * n:] = tanh(gates[:, 3 * n:])
            # This is basically a slicing step
            gf, gi, cand, go = np.transpose(gates.reshape(self.fanin, 4, self.neurons), axes=(1, 0, 2))
            self.states.append(gf * self.states[-1] + gi * cand)
            self.outputs.append(go * tanh(self.states[-1]))
            self.caches.append((gf, gi, cand, go))

        for time in range(self.time):
            timestep(time)

        # gates = X.dot(self.weights) + self.biases
        # gates[:, :n*3] = sigmoid(gates[:, n*3])
        # gates[:, 3 * n:] = tanh(gates[:, 3 * n:])
        # gf, gi, cand, go = np.transpose(gates.reshape(self.fanin, 4, self.neurons), axes=(1, 0, 2))
        #
        # self.output = go * tanh(self.state.top)
        # self.cache = gf, gi, cand, go

    def weight_update(self):
        # Update weights and biases
        np.subtract(self.weights, self.gate_W_gradients * self.brain.eta,
                    out=self.weights)
        np.subtract(self.biases, np.mean(self.error, axis=0) * self.brain.eta,
                    out=self.biases)

    def backpropagation(self):
        error_now = self.error
        error_tomorrow = np.zeros_like(self.error)
        gate_errors = np.zeros((self.brain.m, self.neurons * 4))
        dstate = np.zeros_like(self.states[0])

        # No momentum (yet)
        self.gate_W_gradients = np.zeros_like(self.weights)
        self.gate_b_gradients = np.zeros_like(self.biases)

        # backprop through time
        for t in range(0, self.time, -1):
            gf, gi, cand, go = self.caches[t]
            error_now += error_tomorrow
            dgo = sigmoid.derivative(tanh(self.states[t]) * error_now)
            dstate = tanh.derivative(self.states[t]) * (go * error_now + dstate)
            dgf = sigmoid.derivative(gf) * (self.states[t-1] * dstate)
            dgi = sigmoid.derivative(gi) * (cand * dstate)
            dcand = tanh.derivative(cand) * (gi * dstate)

            gate_errors = np.concatenate((dgf, dgi, dcand, dgo))
            self.gate_W_gradients += self.inputs.T.dot(gate_errors)
            self.gate_b_gradients += gate_errors
            # Folding the (fanin, 4*neurons) part into (4, neurons)
            # then summing the 4 matrices into 1 and getting (fanin, neurons)
            error_tomorrow = np.transpose(gate_errors.reshape(self.fanin, 4, self.neurons), axes=(1, 0, 2)
                                          ).sum(axis=0)
            dstate = gf * dstate

        prev_error = np.dot(gate_errors, self.weights.T)

    def receive_error(self, error_vector: np.ndarray):
        self.error = error_vector


def reference_init():
    H = 128  # Number of LSTM layer's neurons
    D = 12  # Number of input dimension == number of items in vocabulary
    Z = H + D  # Because we will concatenate LSTM state with the input

    model = dict(
        Wf=np.random.randn(Z, H) / np.sqrt(Z / 2.),
        Wi=np.random.randn(Z, H) / np.sqrt(Z / 2.),
        Wc=np.random.randn(Z, H) / np.sqrt(Z / 2.),
        Wo=np.random.randn(Z, H) / np.sqrt(Z / 2.),
        Wy=np.random.randn(H, D) / np.sqrt(D / 2.),
        bf=np.zeros((1, H)),
        bi=np.zeros((1, H)),
        bc=np.zeros((1, H)),
        bo=np.zeros((1, H)),
        by=np.zeros((1, D))
    )
    return model, H, D, Z


def reference_feedforward(X, state):
    model = reference_init()
    m, H, D, Z = model
    Wf, Wi, Wc, Wo, Wy = m['Wf'], m['Wi'], m['Wc'], m['Wo'], m['Wy']
    bf, bi, bc, bo, by = m['bf'], m['bi'], m['bc'], m['bo'], m['by']

    h_old, c_old = state

    # One-hot encode
    X_one_hot = np.zeros(D)
    X_one_hot[X] = 1.
    X_one_hot = X_one_hot.reshape(1, -1)

    # Concatenate old state with current input
    X = np.column_stack((h_old, X_one_hot))

    hf = sigmoid(X @ Wf + bf)
    hi = sigmoid(X @ Wi + bi)
    ho = sigmoid(X @ Wo + bo)
    hc = tanh(X @ Wc + bc)

    c = hf * c_old + hi * hc
    h = ho * tanh(c)

    y = h @ Wy + by
    prob = softmax(y)

    cache = hf, hi, ho, hc, c, c_old, h, m, X, model  # Add all intermediate variables to this cache

    return prob, cache


def reference_backpropagation(m, prob, y_train, d_next, cache):
    # Unpack the cache variables to get the intermediate variables used in forward step

    Wf, Wi, Wc, Wo, Wy = m['Wf'], m['Wi'], m['Wc'], m['Wo'], m['Wy']
    dsigmoid = sigmoid.derivative
    dtanh = tanh.derivative
    hf, hi, ho, hc, c, c_old, h, m, X, (m, H, D, Z) = cache

    dh_next, dc_next = d_next

    # Softmax loss gradient
    dy = prob.copy()
    dy[1, y_train] -= 1.

    # Hidden to output gradient
    dWy = h.T @ dy
    dby = dy
    # Note we're adding dh_next here
    dh = dy @ Wy.T + dh_next

    # Gradient for ho in h = ho * tanh(c)
    dho = tanh(c) * dh
    dho = dsigmoid(dh) * dho

    # Gradient for c in h = ho * tanh(c), note we're adding dc_next here
    dc = ho * dh + dc_next
    dc = dtanh(c) * dc

    # Gradient for hf in c = hf * c_old + hi * hc
    dhf = c_old * dc
    dhf = dsigmoid(hf) * dhf

    # Gradient for hi in c = hf * c_old + hi * hc
    dhi = hc * dc
    dhi = dsigmoid(hi) * dhi

    # Gradient for hc in c = hf * c_old + hi * hc
    dhc = hi * dc
    dhc = dtanh(hc) * dhc

    # Gate gradients, just a normal fully connected layer gradient
    dWf = X.T @ dhf
    dbf = dhf
    dXf = dhf @ Wf.T

    dWi = X.T @ dhi
    dbi = dhi
    dXi = dhi @ Wi.T

    dWo = X.T @ dho
    dbo = dho
    dXo = dho @ Wo.T

    dWc = X.T @ dhc
    dbc = dhc
    dXc = dhc @ Wc.T

    # As X was used in multiple gates, the gradient must be accumulated here
    dX = dXo + dXc + dXi + dXf
    # Split the concatenated X, so that we get our gradient of h_old
    dh_next = dX[:, :H]
    # Gradient for c_old in c = hf * c_old + hi * hc
    dc_next = hf * dc

    grad = dict(Wf=dWf, Wi=dWi, Wc=dWc, Wo=dWo, Wy=dWy, bf=dbf, bi=dbi, bc=dbc, bo=dbo, by=dby)
    state = (dh_next, dc_next)

    return grad, state
