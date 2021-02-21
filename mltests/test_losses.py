#!/usr/bin/env python3

from mltests.includes import *


def test_squared_error(N=5):
    from jagerml.losses import SquaredLoss

    squareError = SquaredLoss()

    torchLoss = nn.MSELoss()
    # input = torch.randn(3, 5, requires_grad=True)
    # target = torch.randn(3, 5)
    # print(torchLoss(input, target))
    # output = torchLoss(input, target)
    # print(output.backward())

    n_dims = np.random.randint(2, 100)
    n_examples = np.random.randint(1, 1000)
    y = y_pred = random_tensor((n_examples, n_dims))

    tf_loss = tf.keras.losses.MSE(y, y_pred)
    print("TF :", tf_loss)

    torchInput = torch.randn(n_examples, n_dims, requires_grad=True)
    torchTarget = torch.randn(n_examples, n_dims)

    # print(squareError.loss(y, y_pred))
    # b = torchLoss(torchInput, torchTarget)
    # print(b.backward())
    # assert_almost_equal(squareError.loss(y, y_pred), torchLoss(y, y_pred))

    input = torch.randn(3, 5, requires_grad=True)
    print(input)
    target = torch.randn(3, 5)
    print(target)
    mse_loss = nn.MSELoss()
    output = mse_loss(input, target)
    print(output)
    output.backward()
    print(output)

    y_true = np.random.randint(0, 2, size=(5, 5))
    print(y_true)
    y_pred = np.random.random(size=(5, 5))
    print(y_pred)
    loss = tf.keras.losses.mean_squared_error(y_true, y_pred)
    print(loss)
    print(np.array_equal(loss.numpy(), np.mean(np.square(y_true - y_pred), axis=-1)))

    print(squareError.loss(y_true, y_pred))


def test_mean_squared_error_v2(N=5):
    from jagerml.losses import MeanSquaredLoss

    i = 1
    while i < N + 1:
        n_dims = np.random.randint(2, 100)
        n_examples = np.random.randint(1, 1000)
        y = stochastic_matrix(n_examples, n_dims)
        y_pred = stochastic_matrix(n_examples, n_dims)

        mse = tf.keras.losses.MeanSquaredError()
        print("TF MSE :", mse(y, y_pred).numpy())

        torchLoss = nn.MSELoss()
        b = torchLoss(torch.tensor(y), torch.tensor(y_pred))
        print("T      :", b.numpy())

        squareError = MeanSquaredLoss()
        print("JAGER  :", squareError(y, y_pred))

        assert_almost_equal(squareError(y, y_pred), b.numpy(), decimal=5)
        assert_almost_equal(squareError(y, y_pred), mse(y, y_pred).numpy(), decimal=5)
        i += 1
    print("Test PASSED")


def test_cross_entropy(N=5):
    from jagerml.losses import CrossEntropy
    from sklearn.metrics import log_loss

    n_classes = np.random.randint(2, 100)
    n_examples = np.random.randint(1, 1000)
    y = one_hot_matrix(n_examples, n_classes)
    y_pred = stochastic_matrix(n_examples, n_classes)

    bc = tf.keras.losses.BinaryCrossentropy()
    # softMax = tf.keras.activations.softmax(bc(y, y_pred).numpy())
    print("TF MSE :", bc(y, y_pred).numpy())

    print("log_loss :", log_loss(y, y_pred))

    # torchLoss = nn.CrossEntropyLoss()
    # b = torchLoss(torch.tensor(y), torch.tensor(y_pred))
    # print("T      :", b.numpy())

    crossEntropy = CrossEntropy()
    print(crossEntropy, crossEntropy(y, y_pred))
    print(crossEntropy, crossEntropy.loss(y, y_pred))
    print(crossEntropy, crossEntropy.loss(y, y_pred))

    # pred1 = crossEntropy.log_softmax(y)
    # loss1 = crossEntropy.nll(pred1, y_pred)
    # print(crossEntropy, loss1)

    #print(tf.nn.softmax_cross_entropy_with_logits(y, y_pred))



if __name__ == "__main__":
    check_version_proxy()
    # test_squared_error()
    # test_mean_squared_error_v2()
    test_cross_entropy()
