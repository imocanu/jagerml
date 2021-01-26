from dataset import *
from prep import prep_normalize, prep_plot_columns, prep_split_data
from window_gen import WindowGen


def run_forecast():
    check_version()
    url = 'https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip'
    fname = 'jena_climate_2009_2016.csv.zip'
    zip_path = dataset_keras_utils_get_file_path(url, fname)

    df = pd.read_csv(zip_path)
    df = df[5::6]

    date_time = pd.to_datetime(df.pop('Date Time'), format='%d.%m.%Y %H:%M:%S')
    print(df.head())
    print(df.describe().transpose())

    ##########################################################################
    wv = df['wv (m/s)']
    bad_wv = wv == -9999.0
    wv[bad_wv] = 0.0

    max_wv = df['max. wv (m/s)']
    bad_max_wv = max_wv == -9999.0
    max_wv[bad_max_wv] = 0.0

    # The above inplace edits are reflected in the DataFrame
    df['wv (m/s)'].min()

    print(df.describe().transpose())

    wv = df.pop('wv (m/s)')
    max_wv = df.pop('max. wv (m/s)')

    # Convert to radians.
    wd_rad = df.pop('wd (deg)') * np.pi / 180
    #
    # # Calculate the wind x and y components.
    df['Wx'] = wv * np.cos(wd_rad)
    df['Wy'] = wv * np.sin(wd_rad)
    #
    # # Calculate the max wind x and y components.
    df['max Wx'] = max_wv * np.cos(wd_rad)
    df['max Wy'] = max_wv * np.sin(wd_rad)
    #######################################################################

    # prep_plot_columns(df, date_time, ['T (degC)', 'p (mbar)', 'rho (g/m**3)'])
    train_df, val_df, test_df = prep_split_data(df)
    train_df, val_df, test_df = prep_normalize(train_df, val_df, test_df, df, plotNorm=False)

    # **********************************************************************
    w1 = WindowGen(input_width=24,
                   label_width=1,
                   shift=24,
                   train_df=train_df,
                   val_df=val_df,
                   test_df=test_df,
                   label_columns=['T (degC)'])
    # print(w1)

    w2 = WindowGen(input_width=6,
                   label_width=1,
                   shift=1,
                   train_df=train_df,
                   val_df=val_df,
                   test_df=test_df,
                   label_columns=['T (degC)'])
    # print(w2)

    # example_window = tf.stack([np.array(train_df[:w2.total_window_size]),
    #                            np.array(train_df[100:100 + w2.total_window_size]),
    #                            np.array(train_df[200:200 + w2.total_window_size])])
    #
    # example_inputs, example_labels = w2.split_window(example_window)
    #
    # w2.example = example_inputs, example_labels
    #
    # print('All shapes are: (batch, time, features)')
    # print(f'Window shape: {example_window.shape}')
    # print(f'Inputs shape: {example_inputs.shape}')
    # print(f'labels shape: {example_labels.shape}')

    # w2.plot()
    # plt.show()

    # print(w2.train.element_spec)

    # **********************************************************************

    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

    # Single step model

    single_step_window = WindowGen(input_width=1,
                                   label_width=1,
                                   shift=1,
                                   train_df=train_df,
                                   val_df=val_df,
                                   test_df=test_df,
                                   label_columns=['T (degC)'])
    print(single_step_window)

    # Baseline model

    single_step_window = WindowGen(input_width=1,
                                   label_width=1,
                                   shift=1,
                                   train_df=train_df,
                                   val_df=val_df,
                                   test_df=test_df,
                                   label_columns=['T (degC)'])
    print(single_step_window)

    wide_window = WindowGen(input_width=24,
                            label_width=24,
                            shift=1,
                            train_df=train_df,
                            val_df=val_df,
                            test_df=test_df,
                            label_columns=['T (degC)'])

    print(wide_window)

    # wide_window.plot(baseline)
    linear = tf.keras.Sequential([
        tf.keras.layers.Dense(units=1)
    ])

    MAX_EPOCHS = 20

    def compile_and_fit(model, window, patience=2):
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                          patience=patience,
                                                          mode='min')

        model.compile(loss=tf.losses.MeanSquaredError(),
                      optimizer=tf.optimizers.Adam(),
                      metrics=[tf.metrics.MeanAbsoluteError()])

        history = model.fit(window.train, epochs=MAX_EPOCHS,
                            validation_data=window.val,
                            callbacks=[early_stopping])
        return history

    history = compile_and_fit(linear, single_step_window)

    val_performance = {}
    performance = {}

    val_performance['Linear'] = linear.evaluate(single_step_window.val)
    performance['Linear'] = linear.evaluate(single_step_window.test, verbose=0)

    # wide_window.plot(linear)
    # plt.show()

    dense = tf.keras.Sequential([
        tf.keras.layers.Dense(units=64, activation='relu'),
        tf.keras.layers.Dense(units=64, activation='relu'),
        tf.keras.layers.Dense(units=1)
    ])

    history = compile_and_fit(dense, single_step_window)

    val_performance['Dense'] = dense.evaluate(single_step_window.val)
    performance['Dense'] = dense.evaluate(single_step_window.test, verbose=0)

    CONV_WIDTH = 3
    conv_window = WindowGen(
        input_width=CONV_WIDTH,
        label_width=1,
        shift=1,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        label_columns=['T (degC)'])

    multi_step_dense = tf.keras.Sequential([
        # Shape: (time, features) => (time*features)
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=32, activation='relu'),
        tf.keras.layers.Dense(units=32, activation='relu'),
        tf.keras.layers.Dense(units=1),
        # Add back the time dimension.
        # Shape: (outputs) => (1, outputs)
        tf.keras.layers.Reshape([1, -1]),
    ])

    history = compile_and_fit(multi_step_dense, conv_window)

    val_performance['Multi step dense'] = multi_step_dense.evaluate(conv_window.val)
    performance['Multi step dense'] = multi_step_dense.evaluate(conv_window.test, verbose=0)

    conv_model = tf.keras.Sequential([
        tf.keras.layers.Conv1D(filters=32,
                               kernel_size=(CONV_WIDTH,),
                               activation='relu'),
        tf.keras.layers.Dense(units=32, activation='relu'),
        tf.keras.layers.Dense(units=1),
    ])

    history = compile_and_fit(conv_model, conv_window)

    val_performance['Conv'] = conv_model.evaluate(conv_window.val)
    performance['Conv'] = conv_model.evaluate(conv_window.test, verbose=0)





    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@


if __name__ == "__main__":
    run_forecast()
