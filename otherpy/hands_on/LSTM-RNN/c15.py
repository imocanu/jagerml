from dataset import *


N_STEPS = 50

def plot_series(series, y=None, y_pred=None, x_label="$t$", y_label="$x(t)$"):
    plt.plot(series, ".-")
    if y is not None:
        plt.plot(N_STEPS, y, "bx", markersize=10)
    if y_pred is not None:
        plt.plot(N_STEPS, y_pred, "ro")
    plt.grid(True)
    if x_label:
        plt.xlabel(x_label, fontsize=16)
    if y_label:
        plt.ylabel(y_label, fontsize=16, rotation=0)
    plt.hlines(0, 0, 100, linewidth=1)
    plt.axis([0, N_STEPS + 1, -1, 1])

def run_lstm():
    check_version()
    series = generate_time_series(10000, N_STEPS + 1)
    X_train, y_train = series[:7000, :N_STEPS], series[:7000, -1]
    X_valid, y_valid = series[7000:9000, :N_STEPS], series[7000:9000, -1]
    X_test, y_test = series[9000:, :N_STEPS], series[9000:, -1]
    print(X_train.shape, y_train.shape)

    fig, axes = plt.subplots(nrows=1, ncols=3, sharey=True, figsize=(12, 4))
    for col in range(3):
        plt.sca(axes[col])
        plot_series(X_valid[col, :, 0], y_valid[col, 0],
                    y_label=("$x(t)$" if col == 0 else None))
    # save_fig("time_series_plot")
    plt.show()

    y_pred = X_valid[:, -1]
    print(np.mean(keras.losses.mean_squared_error(y_valid, y_pred)))

    # plot_series(X_valid[0, :, 0], y_valid[0, 0], y_pred[0, 0])
    # plt.show()

    model = keras.models.Sequential([
        keras.layers.Flatten(input_shape=[50, 1]),
        keras.layers.Dense(1)
    ])

    model.compile(loss="mse", optimizer="adam")
    history = model.fit(X_train, y_train, epochs=20,
                        validation_data=(X_valid, y_valid))

    model.evaluate(X_valid, y_valid)

    # def plot_learning_curves(loss, val_loss):
    #     plt.plot(np.arange(len(loss)) + 0.5, loss, "b.-", label="Training loss")
    #     plt.plot(np.arange(len(val_loss)) + 1, val_loss, "r.-", label="Validation loss")
    #     plt.gca().xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
    #     plt.axis([1, 20, 0, 0.05])
    #     plt.legend(fontsize=14)
    #     plt.xlabel("Epochs")
    #     plt.ylabel("Loss")
    #     plt.grid(True)

    y_pred = model.predict(X_valid)
    plot_series(X_valid[0, :, 0], y_valid[0, 0], y_pred[0, 0])
    plt.show()

def run_rnn():
    check_version()
    series = generate_time_series(10000, N_STEPS + 1)
    X_train, y_train = series[:7000, :N_STEPS], series[:7000, -1]
    X_valid, y_valid = series[7000:9000, :N_STEPS], series[7000:9000, -1]
    X_test, y_test = series[9000:, :N_STEPS], series[9000:, -1]
    print(X_train.shape, y_train.shape)

    model = keras.models.Sequential([
        keras.layers.SimpleRNN(1, input_shape=[None, 1])
    ])

    optimizer = keras.optimizers.Adam(lr=0.005)
    model.compile(loss="mse", optimizer=optimizer)
    history = model.fit(X_train, y_train, epochs=20,
                        validation_data=(X_valid, y_valid))

    model.evaluate(X_valid, y_valid)

    model = keras.models.Sequential([
        keras.layers.SimpleRNN(20, return_sequences=True, input_shape=[None, 1]),
        keras.layers.SimpleRNN(20, return_sequences=True),
        keras.layers.SimpleRNN(1)
    ])

    model.compile(loss="mse", optimizer="adam")
    history = model.fit(X_train, y_train, epochs=20,
                        validation_data=(X_valid, y_valid))

def run_forecast():
    check_version()
    series = generate_time_series(1, N_STEPS + 10)
    X_new, Y_new = series[:, :N_STEPS], series[:, N_STEPS:]
    X = X_new
    for step_ahead in range(10):
        y_pred_one = model.predict(X[:, step_ahead:])[:, np.newaxis, :]
        X = np.concatenate([X, y_pred_one], axis=1)

    Y_pred = X[:, N_STEPS:]


if __name__ == "__main__":
    # run_lstm()
    # run_rnn()
    run_forecast()

