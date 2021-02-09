#!/usr/bin/env python3

from mltests.includes import *


def torch_gradient_generator(fn, **kwargs):
    def get_grad(z):
        z1 = torch.autograd.Variable(torch.from_numpy(z), requires_grad=True)
        z2 = fn(z1, **kwargs).sum()
        z2.backward()
        grad = z1.grad.numpy()
        return grad

    return get_grad


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


def test_leaky_relu(loops=100):
    from jagerml.activations import LeakyReLU
    print("[*] LeakyReLU : forward & backward")
    start = time.time()

    for _ in range(loops):
        features = np.random.randint(100, 1000)
        labels = np.random.randint(100, 1000)
        alpha = np.random.uniform(0, 100)
        sm = stochastic_matrix(features, labels)
        rt = random_tensor((features, labels))

        leaky_relu = LeakyReLU(alpha=alpha)
        leaky_relu.forward(sm, None)
        leaky_relu.backward(rt)
        torch_test = F.leaky_relu(torch.FloatTensor(sm), alpha).numpy()
        tf_test = tf.nn.leaky_relu(tf.convert_to_tensor(sm), alpha).numpy()
        assert_almost_equal(leaky_relu.output, tf_test)
        assert_almost_equal(leaky_relu.output, torch_test)

        torch_test = torch_gradient_generator(F.leaky_relu,
                                              negative_slope=alpha)
        assert_almost_equal(leaky_relu.dinputs, torch_test(rt), decimal=6)

    end = time.time()
    print("test pass in {:0.2f} s".format(end - start))


def test_leaky_relu_base(loops=100):
    from jagerml.activations import LeakyReLUbase
    print("[*] LeakyReLU : forward & backward")
    start = time.time()

    for _ in range(loops):
        features = np.random.randint(100, 1000)
        labels = np.random.randint(100, 1000)
        alpha = np.random.uniform(0, 100)
        sm = stochastic_matrix(features, labels)
        rt = random_tensor((features, labels))

        leaky_relu = LeakyReLUbase(alpha=alpha)
        torch_test = F.leaky_relu(torch.FloatTensor(sm), alpha).numpy()
        tf_test = tf.nn.leaky_relu(tf.convert_to_tensor(sm), alpha).numpy()
        assert_almost_equal(leaky_relu.fn(sm), tf_test)
        assert_almost_equal(leaky_relu.fn(sm), torch_test)

        torch_test = torch_gradient_generator(F.leaky_relu,
                                              negative_slope=alpha)
        assert_almost_equal(leaky_relu.grad(rt), torch_test(rt), decimal=6)

    end = time.time()
    print("test pass in {:0.2f} s".format(end - start))


def test_relu(loops=100):
    from jagerml.activations import ReLU
    print("[*] ReLU")
    start = time.time()
    for _ in range(loops):
        features = np.random.randint(100, 1000)
        labels = np.random.randint(100, 1000)
        sm = stochastic_matrix(features, labels)

        relu = ReLU()
        relu.forward(sm, None)
        torch_test = F.relu(torch.FloatTensor(sm)).numpy()
        tf_test = tf.nn.relu(tf.convert_to_tensor(sm)).numpy()
        assert_almost_equal(relu.output, tf_test)
        assert_almost_equal(relu.output, torch_test)
    end = time.time()
    print("test pass in {:0.2f} s".format(end - start))


def test_relubase(loops=100):
    from jagerml.activations import ReLUbase
    print("[*] ReLUbase")
    start = time.time()
    for _ in range(loops):
        features = np.random.randint(100, 1000)
        labels = np.random.randint(100, 1000)
        sm = stochastic_matrix(features, labels)

        relu = ReLUbase()
        output = relu.fn(sm)
        torch_test = F.relu(torch.FloatTensor(sm)).numpy()
        tf_test = tf.nn.relu(tf.convert_to_tensor(sm)).numpy()
        assert_almost_equal(output, tf_test)
        assert_almost_equal(output, torch_test)
    end = time.time()
    print("test pass in {:0.2f} s".format(end - start))


def test_softmaxbase(loops=100):
    from jagerml.activations import Softmaxbase
    print("[*] Softmaxbase")
    start = time.time()
    for _ in range(loops):
        features = np.random.randint(100, 1000)
        labels = np.random.randint(100, 1000)
        sm = stochastic_matrix(features, labels)

        softmax = Softmaxbase()
        output = softmax.fn(sm)
        torch_test = F.softmax(torch.FloatTensor(sm), dim=1).numpy()
        tf_test = tf.nn.softmax(tf.convert_to_tensor(sm)).numpy()
        assert_almost_equal(output, tf_test)
        assert_almost_equal(output, torch_test)
    end = time.time()
    print("test pass in {:0.2f} s".format(end - start))


def test_linearbase(loops=100):
    from jagerml.activations import Linearbase
    print("[*] Test Linear activator")
    start = time.time()
    for i in range(loops):
        features = np.random.randint(1, 1000)
        labels = np.random.randint(1, 1000)
        sm = stochastic_matrix(features, labels)
        rt = random_tensor((features, labels))

        linear = Linearbase(input=rt, weight=sm)
        output = linear.fn(sm)

        torch_test = F.linear(torch.FloatTensor(rt),
                              torch.FloatTensor(sm))
        assert_almost_equal(output, torch_test.numpy(), decimal=3)
        print("[>] trial {:>3} features {:>4} labels {:>4}".format(i+1,
                                                                   features,
                                                                   labels))

    end = time.time()
    print("Test PASS in {:0.2f} s".format(end - start))


def test_activations():
    test_linearbase(5)
    test_leaky_relu(5)
    test_leaky_relu_base(5)
    test_relu(5)
    test_relubase(5)
    test_softmaxbase(5)


if __name__ == "__main__":
    test_activations()
