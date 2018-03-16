import numpy as np
import scipy.io

from enum import Enum

from memory_profiler import profile

from NN import NN


def load_dataset_from_mat_file(mat_file):
    mat = scipy.io.loadmat(mat_file)
    X = mat['X']
    y = mat['y']

    Y = np.ndarray(shape=(len(y), 10), dtype=float, order='F')
    for i in range(0, len(y)):
        yi = np.zeros(10)
        yi[y[i] - 1] = 1
        Y[i] = yi

    return X, Y


class TrainType(Enum):
    COMMON = 0
    MEM_OPT = 1


class TrainNN:

    layer_size = 4000
    hidden_size = 5
    learning_rate = 1e-2

    @profile
    def __init__(self, x, y, layer_size=4000):
        self.x = x
        self.y = y
        self.in_size = self.x[-1].size
        self.out_size = self.y[-1].size
        self.layer_size = layer_size
        self.__init_nn()


    def calc_delta(self, delta_next_values, cur_outs, cur_weights):

        l_cur = len(cur_outs)
        deltas = np.zeros(l_cur)

        for i in range(0, l_cur):
            weights_from_i_to_layer = cur_weights[i]
            cur_out = cur_outs[i]

            sum = 0.0
            l_next = len(delta_next_values)
            for j in range(0, l_next):
                sum += delta_next_values[j] * weights_from_i_to_layer[j]

            deltas[i] = sum * cur_out * (1 - cur_out) / len(self.x)

        return deltas



    def backward_separate_layers(self, outs, next_deltas, layer_from, layer_to):

        i = layer_from - 1
        j = len(outs) - 2

        while i >= layer_to and j >= 0:
            cur_outs = outs[j]
            cur_weights = self.nn.weights_list[i]

            # update weights
            grad_weights = cur_outs.T.dot(next_deltas)
            cur_weights += self.learning_rate * grad_weights

            cur_deltas = []
            batch_size = len(cur_outs)
            for k in range(0, batch_size):
                cur_deltas.append(self.calc_delta(next_deltas[k], cur_outs[k], cur_weights))

            next_deltas = cur_deltas

            i -= 1
            j -= 1

        return next_deltas



    def backward_common(self, all_outs):
        next_deltas = self.__calc_deltas_for_output_layer(all_outs[-1])
        self.backward_separate_layers(all_outs, next_deltas, self.layer_size - 1, 0)

    @profile
    def train(self, max_iter, type=TrainType.COMMON):

        prev_loss = float('inf')
        prev_weights = None
        for i in range(0, max_iter):

            if type == TrainType.MEM_OPT:
                self.forward_and_back_with_memory_optimization()
            else:
                self.forward_and_backward()

            print(self.loss)

            # if self.loss > prev_loss:
            #     return prev_weights
            #
            # prev_weights = self.nn.weights_list.copy()

        return self.nn.weights_list


    @profile
    def forward_and_backward(self):
        all_out = self.nn.forward_and_memorize_all_outs(self.x)
        self.backward_common(all_out)


    @profile
    def forward_and_back_with_memory_optimization(self):
        size = TrainNN.__get_sqrt_array_size(self.layer_size)

        layer_indices = []
        i = 0
        while i < self.layer_size:
            layer_indices.append(i)
            i += size

        outs = self.nn.forward_and_get_outs_in_indices(self.x, layer_indices)

        next_layer = self.layer_size - 1
        i = len(layer_indices) - 1 if layer_indices[-1] != self.layer_size - 1 else len(layer_indices) - 2


        last_delta_setup = False
        while i >= 0:
            temp_outs = self.nn.forward_separate_layer(outs[i], layer_indices[i], next_layer + 1)

            #TODO: refactor it
            if last_delta_setup is False:
                next_deltas = self.__calc_deltas_for_output_layer(temp_outs[-1])
                last_delta_setup = True

            next_deltas = self.backward_separate_layers(temp_outs, next_deltas, next_layer, layer_indices[i])
            next_layer = layer_indices[i]

            i -= 1



    @profile
    def __init_nn(self):

        weights_list = []
        in_size = self.in_size
        for i in range(0, self.layer_size - 2):
            out_size = self.hidden_size
            wi = np.random.rand(in_size, out_size)
            weights_list.append(wi)
            in_size = out_size

        out_size = self.out_size
        w_last = np.random.rand(in_size, out_size)
        weights_list.append(w_last)

        self.nn = NN(weights_list)

    def __calc_deltas_for_output_layer(self, out):

        self.loss = self.__calc_loss(out)

        error = out - self.y
        out_der = out * (1 - out)

        deltas = error * out_der
        deltas = deltas / len(self.x)
        return deltas

    def __calc_loss(self, target):
        loss = np.square(target, self.y).sum()
        return loss

    @staticmethod
    def __get_sqrt_array_size(n):
        sqrt_n = int(np.sqrt(n))
        if sqrt_n ** 2 == n:
            return sqrt_n

        return sqrt_n + 1


x, y = load_dataset_from_mat_file('data/ex4data.mat')

trainNN = TrainNN(x, y)
weights_list = trainNN.train(100, TrainType.MEM_OPT)
np.save('weight.npy', weights_list)


#n = 100
#for i in range(0, 100):
#    trainNN = TrainNN(x, y, n)
#    weights_list = trainNN.train(1, TrainType.MEM_OPT)
#    n += 200















