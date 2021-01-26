from dataset import *


# def prep_pandas(csv_path=None):
#     if csv_path is None:
#         print("csv path is None !")
#     return None
#
#     df = pd.read_csv(csv_path)
#     return df

def prep_plot_columns(df, date_time, plot_cols=[]):
    plot_features = df[plot_cols]
    plot_features.index = date_time
    _ = plot_features.plot(subplots=True)

    plot_features = df[plot_cols][:480]
    plot_features.index = date_time[:480]
    _ = plot_features.plot(subplots=True)
    plt.show()


def prep_split_data(df, train_size=0.7, val_size=0.9, test_size=0.9):
    column_indices = {name: i for i, name in enumerate(df.columns)}
    n = len(df)
    train_df = df[0:int(n * train_size)]
    val_df = df[int(n * train_size):int(n * val_size)]
    test_df = df[int(n * test_size):]
    num_features = df.shape[1]
    print("[*] Total features :", num_features)
    print("[*] Total inputs   :", df.shape[0])

    return train_df, val_df, test_df


def prep_normalize(train_df, val_df, test_df, df, plotNorm=False):
    train_mean = train_df.mean()
    train_std = train_df.std()

    train_df = (train_df - train_mean) / train_std
    val_df = (val_df - train_mean) / train_std
    test_df = (test_df - train_mean) / train_std

    if plotNorm:
        df_std = (df - train_mean) / train_std
        df_std = df_std.melt(var_name='Column', value_name='Normalized')
        plt.figure(figsize=(12, 6))
        ax = sns.violinplot(x='Column', y='Normalized', data=df_std)
        _ = ax.set_xticklabels(df.keys(), rotation=90)
        plt.show()

    return train_df, val_df, test_df