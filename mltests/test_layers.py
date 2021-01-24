#!/usr/bin/env python3

from mltests.includes import *


def random_tensor(shape, standardize=False):
    offset = np.random.randint(-300, 300, shape)
    X = np.random.rand(*shape) + offset

    if standardize:
        eps = np.finfo(float).eps
        X = (X - X.mean(axis=0)) / (X.std(axis=0) + eps)
    return X


def stochastic_matrix(features, labels):
    sm = np.random.rand(features, labels)
    sm /= sm.sum(axis=1, keepdims=True)
    return sm


def generate_gray_image():
    # Simulate a 28 x 28 pixel, grayscale "image"
    batch_size = 32
    input = torch.randn(1, 28, 28)
    print(input.shape)
    # input = input.view(batch_size, -1)
    # print(input.shape)
    return input


def err_fmt(params, golds, ix, warn_str=""):
    mine, label = params[ix]
    err_msg = "-" * 25 + " DEBUG " + "-" * 25 + "\n"
    prev_mine, prev_label = params[max(ix - 1, 0)]
    err_msg += "Mine (prev) [{}]:\n{}\n\nTheirs (prev) [{}]:\n{}".format(
        prev_label, prev_mine, prev_label, golds[prev_label]
    )
    err_msg += "\n\nMine [{}]:\n{}\n\nTheirs [{}]:\n{}".format(
        label, mine, label, golds[label]
    )
    err_msg += warn_str
    err_msg += "\n" + "-" * 23 + " END DEBUG " + "-" * 23
    return err_msg


class TorchLinearActivation(nn.Module):
    def __init__(self):
        super(TorchLinearActivation, self).__init__()
        pass

    @staticmethod
    def forward(input):
        return input

    @staticmethod
    def backward(grad_output):
        return torch.ones_like(grad_output)


@pytest.fixture
def iris_data():
    return datasets.load_iris().data


@pytest.mark.parametrize("inputs, neurons", [
    (4, 1),
    (4, 10),
    (4, 100),
    (4, 1000),
])
def test_simple_dense(inputs, neurons, iris_data):
    dense = Dense(inputs, neurons)
    assert dense.forward(iris_data) == True


def test_DenseLayer(N=1):
    from jagerml.layers import DenseLayer
    from jagerml.activations import LeakyReLUbase

    np.random.seed(12345)

    N = np.inf if N is None else N

    i = 1
    while i < N + 1:
        n_ex = np.random.randint(1, 100)
        n_in = np.random.randint(1, 100)
        n_out = np.random.randint(1, 100)
        X = random_tensor((n_ex, n_in), standardize=True)

        # randomly select an activation function
        act_fn = None
        torch_fn = TorchLinearActivation()
        act_fn_name = "Linearbase"

        # initialize FC layer
        L1 = DenseLayer(n_out=n_out, act_fn=None)
        # print(L1.hyperparameters)

        # # forward prop
        y_pred = L1.forward(X)
        #
        # # backprop
        dLdy = np.ones_like(y_pred)
        dLdX = L1.backward(dLdy)
        #
        # # get gold standard gradients
        gold_mod = TorchDenseLayer(n_in, n_out, torch_fn, L1.parameters)
        golds = gold_mod.extract_grads(X)

        params = [
            (L1.X[0], "X"),
            (y_pred, "y"),
            (L1.parameters["W"].T, "W"),
            (L1.parameters["b"], "b"),
            (dLdy, "dLdy"),
            (L1.gradients["W"].T, "dLdW"),
            (L1.gradients["b"], "dLdB"),
            (dLdX, "dLdX"),
        ]

        print("\nTrial {}\nact_fn={}".format(i, act_fn_name))
        for ix, (mine, label) in enumerate(params):
            assert_almost_equal(mine, golds[label], decimal=3)
            print("\tPASSED {}".format(label))
        i += 1


def test_linear_layer():
    features = np.random.randint(1, 1000)
    labels = np.random.randint(1, 1000)
    alpha = np.random.randint(0, 100)
    sm = stochastic_matrix(features, labels)
    rt = random_tensor((labels, features))
    torch_rt = torch.autograd.Variable(torch.FloatTensor(rt),
                                       requires_grad=True)


    fc = nn.Linear(in_features=features, out_features=labels)
    torch_output = fc.forward(torch_rt)
    print("TORCH :\n", torch_output, type(torch_output))

    from jagerml.layers import DenseLayer
    dense = DenseLayer(n_out=labels, n_in=features)
    output = dense.forward(rt)
    print("\n\n")
    print("DENSE :\n", output, type(output))

    # assert_almost_equal(dense.output, torch_output.detach().numpy(), decimal=6)


if __name__ == "__main__":
    test_linear_layer()
    test_DenseLayer()
