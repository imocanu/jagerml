from to_import import *


def run_test():
    """
    https://www.kaggle.com/shahules/basic-eda-cleaning-and-glove
    """
    dataset_path = "../../../../../../datasets/nlp-getting-started/"
    print(os.listdir(dataset_path))
    tweet = pd.read_csv(dataset_path + "train.csv")
    test = pd.read_csv(dataset_path + "test.csv")

    print(tweet.head(3))
    print(test.head(3))
    print(type(tweet), type(test.shape))
    print(tweet.shape[0], tweet.shape[1])
    print(test.shape[0], test.shape[1])
    print(tweet.keys())
    # x = tweet.target.value_counts()
    # sns.barplot(x.index, x)
    # plt.show()

    tweet_len = tweet[tweet['target'] == 1]['text'].str.len()

    tweet_len = tweet[tweet['target'] == 1]['text'].str.split().map(lambda x: len(x))
    q = "Our Deeds are the Reason of this #earthquake May ALLAH Forgive us all"
    test_len = q.split()
    print(q)
    print(test_len)
    print(tweet['text'][:1].to_numpy())
    print(tweet_len[:1])
    print(tweet_len.max())

    word = tweet[tweet['target'] == 1]['text'].str.split().apply(lambda x: [len(i) for i in x])
    print(word.map(lambda x: np.mean(x)))

    def create_corpus(target):
        corpus = []
        for x in tweet[tweet['target'] == target]['text'].str.split():
            for i in x:
                corpus.append(i)
        return corpus

    corpus = create_corpus(0)

    from collections import defaultdict
    from collections import Counter
    from nltk.corpus import stopwords
    from nltk.util import ngrams
    stop = set(stopwords.words('english'))
    dic = defaultdict(int)
    for word in corpus:
        if word in stop:
            dic[word] += 1

    top = sorted(dic.items(), key=lambda x: x[1], reverse=True)[:20]
    print(top)

    x, y = zip(*top)
    # plt.bar(x, y)
    # plt.show()

    # Punctuation
    corpus = create_corpus(1)
    dic = defaultdict(int)
    import string
    special = string.punctuation
    for i in (corpus):
        if i in special:
            dic[i] +=1

    x,y = zip(*dic.items())
    # plt.bar(x, y)
    # plt.show()

    # Most common words
    corpus = create_corpus(0)
    x = []
    y = []
    counter = Counter(corpus)
    most = counter.most_common()
    for word, count in most[:40]:
        if word not in stop:
            x.append(word)
            y.append(count)

    # sns.barplot(x=y, y=x)
    # plt.show()

    # NGRAM




if __name__ == "__main__":
    check_version_proxy_gpu()
    run_test()
