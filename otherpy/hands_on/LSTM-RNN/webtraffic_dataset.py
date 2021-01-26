from dataset import *
from prep import prep_normalize, prep_plot_columns, prep_split_data
from window_gen import WindowGen


def run_forecast():
    check_version()
    url = "/home/imocanu/Music/web-traffic-time-series-forecasting/"
    key_1 = 'key_1.csv'
    key_2 = 'key_2.csv'
    sample_submission_1 = "sample_submission_1.csv"
    sample_submission_2 = "sample_submission_2.csv"
    train_1 = "train_1.csv"
    train_2 = "train_2.csv"


    # df_key_1 = pd.read_csv(url+key_1)
    # df_key_2 = pd.read_csv(url+key_2)
    # df_sample_1 = pd.read_csv(url+sample_submission_1)
    # df_sample_2 = pd.read_csv(url+sample_submission_2)
    df_train_1 = pd.read_csv(url+train_1)
    df_train_2 = pd.read_csv(url+train_2)

    # print(df_key_1.head())
    # print(df_key_2.head())
    # print(df_sample_1.head())
    # print(df_sample_2.head())
    print(df_train_1.head().transpose())
    print(df_train_2.head().transpose())


if __name__ == "__main__":
    run_forecast()
