from dataset import *
from window_gen import *


def split_window(self, features):
    inputs = features[:, self.input_slice, :]
    labels = features[:, self.labels_slice, :]
    if self.label_columns is not None:
        labels = tf.stack(
            [labels[:, :, self.column_indices[name]] for name in self.label_columns],
            axis=-1)

    # Slicing doesn't preserve static shape information, so set the shapes
    # manually. This way the `tf.data.Datasets` are easier to inspect.
    inputs.set_shape([None, self.input_width, None])
    labels.set_shape([None, self.label_width, None])

    return inputs, labels


def plot(self, model=None, plot_col='T (degC)', max_subplots=3):
    inputs, labels = self.example
    plt.figure(figsize=(12, 8))
    plot_col_index = self.column_indices[plot_col]
    max_n = min(max_subplots, len(inputs))
    for n in range(max_n):
        plt.subplot(3, 1, n + 1)
        plt.ylabel(f'{plot_col} [normed]')
        plt.plot(self.input_indices, inputs[n, :, plot_col_index],
                 label='Inputs', marker='.', zorder=-10)

        if self.label_columns:
            label_col_index = self.label_columns_indices.get(plot_col, None)
        else:
            label_col_index = plot_col_index

        if label_col_index is None:
            continue

        plt.scatter(self.label_indices, labels[n, :, label_col_index],
                    edgecolors='k', label='Labels', c='#2ca02c', s=64)
        if model is not None:
            predictions = model(inputs)
            plt.scatter(self.label_indices, predictions[n, :, label_col_index],
                        marker='X', edgecolors='k', label='Predictions',
                        c='#ff7f0e', s=64)

        if n == 0:
            plt.legend()

    plt.xlabel('Time [h]')


def make_dataset(self, data):
    data = np.array(data, dtype=np.float32)
    ds = tf.keras.preprocessing.timeseries_dataset_from_array(
        data=data,
        targets=None,
        sequence_length=self.total_window_size,
        sequence_stride=1,
        shuffle=True,
        batch_size=32, )

    ds = ds.map(self.split_window)

    return ds


class Baseline(tf.keras.Model):
  def __init__(self, label_index=None):
    super().__init__()
    self.label_index = label_index

  def call(self, inputs):
    if self.label_index is None:
      return inputs
    result = inputs[:, :, self.label_index]
    return result[:, :, tf.newaxis]


def run_forecast():
    check_version()
    url = 'https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip'
    fname = 'jena_climate_2009_2016.csv.zip'
    zip_path = dataset_keras_utils_get_file_path(url=url, fname=fname)

    csv_path, _ = os.path.splitext(zip_path)
    print(zip_path)
    print(csv_path)
    df = pd.read_csv(csv_path)
    print(df.head())
    df = df[5::6]
    print(df.head())
    date_time = pd.to_datetime(df.pop('Date Time'), format='%d.%m.%Y %H:%M:%S')
    print(df.head())
    # print(date_time.head())

    # plot_cols = ['T (degC)', 'p (mbar)', 'rho (g/m**3)']
    # plot_features = df[plot_cols]
    # plot_features.index = date_time
    # _ = plot_features.plot(subplots=True)
    #
    # plot_features = df[plot_cols][:480]
    # plot_features.index = date_time[:480]
    # _ = plot_features.plot(subplots=True)
    # plt.show()

    print(df.describe().transpose())

    wv = df['wv (m/s)']
    bad_wv = wv == -9999.0
    wv[bad_wv] = 0.0

    max_wv = df['max. wv (m/s)']
    bad_max_wv = max_wv == -9999.0
    max_wv[bad_max_wv] = 0.0

    # The above inplace edits are reflected in the DataFrame
    df['wv (m/s)'].min()

    wv = df.pop('wv (m/s)')
    max_wv = df.pop('max. wv (m/s)')

    # Convert to radians.
    wd_rad = df.pop('wd (deg)') * np.pi / 180

    # Calculate the wind x and y components.
    df['Wx'] = wv * np.cos(wd_rad)
    df['Wy'] = wv * np.sin(wd_rad)

    # Calculate the max wind x and y components.
    df['max Wx'] = max_wv * np.cos(wd_rad)
    df['max Wy'] = max_wv * np.sin(wd_rad)

    timestamp_s = date_time.map(datetime.datetime.timestamp)
    day = 24 * 60 * 60
    year = 365.2425 * day

    df['Day sin'] = np.sin(timestamp_s * (2 * np.pi / day))
    df['Day cos'] = np.cos(timestamp_s * (2 * np.pi / day))
    df['Year sin'] = np.sin(timestamp_s * (2 * np.pi / year))
    df['Year cos'] = np.cos(timestamp_s * (2 * np.pi / year))

    fft = tf.signal.rfft(df['T (degC)'])
    f_per_dataset = np.arange(0, len(fft))

    n_samples_h = len(df['T (degC)'])
    hours_per_year = 24 * 365.2524
    years_per_dataset = n_samples_h / (hours_per_year)

    f_per_year = f_per_dataset / years_per_dataset
    # plt.step(f_per_year, np.abs(fft))
    # plt.xscale('log')
    # plt.ylim(0, 400000)
    # plt.xlim([0.1, max(plt.xlim())])
    # plt.xticks([1, 365.2524], labels=['1/Year', '1/day'])
    # _ = plt.xlabel('Frequency (log scale)')
    # plt.show()

    column_indices = {name: i for i, name in enumerate(df.columns)}

    n = len(df)
    train_df = df[0:int(n * 0.7)]
    val_df = df[int(n * 0.7):int(n * 0.9)]
    test_df = df[int(n * 0.9):]

    num_features = df.shape[1]

    train_mean = train_df.mean()
    train_std = train_df.std()

    train_df = (train_df - train_mean) / train_std
    val_df = (val_df - train_mean) / train_std
    test_df = (test_df - train_mean) / train_std

    # df_std = (df - train_mean) / train_std
    # df_std = df_std.melt(var_name='Column', value_name='Normalized')
    # plt.figure(figsize=(12, 6))
    # ax = sns.violinplot(x='Column', y='Normalized', data=df_std)
    # _ = ax.set_xticklabels(df.keys(), rotation=90)
    # plt.show()

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

    WindowGen.split_window = split_window

    example_window = tf.stack([np.array(train_df[:w2.total_window_size]),
                               np.array(train_df[100:100 + w2.total_window_size]),
                               np.array(train_df[200:200 + w2.total_window_size])])

    example_inputs, example_labels = w2.split_window(example_window)

    print('All shapes are: (batch, time, features)')
    print(f'Window shape: {example_window.shape}')
    print(f'Inputs shape: {example_inputs.shape}')
    print(f'labels shape: {example_labels.shape}')

    w2.example = example_inputs, example_labels

    WindowGen.plot = plot

    # w2.plot()
    # w2.plot(plot_col='p (mbar)')
    # plt.show()

    WindowGen.make_dataset = make_dataset

    @property
    def train(self):
        return self.make_dataset(self.train_df)

    @property
    def val(self):
        return self.make_dataset(self.val_df)

    @property
    def test(self):
        return self.make_dataset(self.test_df)

    @property
    def example(self):
        """Get and cache an example batch of `inputs, labels` for plotting."""
        result = getattr(self, '_example', None)
        if result is None:
            # No example batch was found, so get one from the `.train` dataset
            result = next(iter(self.train))
            # And cache it for next time
            self._example = result
        return result

    WindowGen.train = train
    WindowGen.val = val
    WindowGen.test = test
    WindowGen.example = example

    print(w2.train.element_spec)

    single_step_window = WindowGen(
        input_width=1,
        label_width=1,
        shift=1,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        label_columns=['T (degC)'])

    print(single_step_window)

    for example_inputs, example_labels in w2.train.take(1):
        print(f'Inputs shape (batch, time, features): {example_inputs.shape}')
        print(f'Labels shape (batch, time, features): {example_labels.shape}')

    baseline = Baseline(label_index=column_indices['T (degC)'])

    baseline.compile(loss=tf.losses.MeanSquaredError(),
                     metrics=[tf.metrics.MeanAbsoluteError()])

    val_performance = {}
    performance = {}

    val_performance['Baseline'] = baseline.evaluate(single_step_window.val)
    performance['Baseline'] = baseline.evaluate(single_step_window.test, verbose=0)


if __name__ == "__main__":
    run_forecast()
