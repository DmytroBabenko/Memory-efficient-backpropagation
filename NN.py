import numpy as np

def sigmoid(x):
    return 1 / ( 1 + np.exp(x))


class NN:

    weights_list = None
    layer_size = 0

    def __init__(self, weights_list):
        self.weights_list = weights_list
        self.layer_size = len(weights_list) + 1


    def forward_and_memorize_all_outs(self, x):
        return self.forward_separate_layer(x, 0, self.layer_size)

    def forward(self, x):
        g = x
        for i in range(0, self.layer_size - 1):
            g = self.__forward(g, self.weights_list[i])
        return g

    def forward_separate_layer(self, x, layer_from, layer_to):
        outs = []
        g = x
        outs.append(g)
        for i in range(layer_from, layer_to - 1):
            g = self.__forward(g, self.weights_list[i])
            outs.append(g)

        return outs

    def forward_and_get_outs_in_indices(self, x, layer_indices):

        if len(layer_indices) == 0:
            return None

        outs = []
        g = x

        j = 0
        if layer_indices[j] == 0:
            outs.append(g)
            j += 1

        i = 0
        while i < self.layer_size - 2 and j < len(layer_indices):
            g = self.__forward(g, self.weights_list[i])

            if i + 1 == layer_indices[j]:
                outs.append(g)
                j += 1
                if j >= len(layer_indices):
                    break

            i += 1

        return outs


    @staticmethod
    def __forward(g_prev, w):
        h = g_prev.dot(w)
        g = sigmoid(h)
        return g
