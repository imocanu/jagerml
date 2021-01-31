from to_import import *


def run_transformers():
    dataset_name = "ted_hrlr_translate/pt_to_en"
    dataset, info = tfds.load(dataset_name,
                              with_info=True,
                              as_supervised=True)

    check_dataset(dataset)

    train_ds, test_ds = dataset['train'], dataset['test']

    def tokenize(lang):
        lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
        lang_tokenizer.fit_on_texts(lang)
        tensor = lang_tokenizer.texts_to_sequences(lang)
        tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor,
                                                               padding='post')
        return tensor, lang_tokenizer

    import unicodedata
    def unicode_to_ascii(s):
        return ''.join(c for c in unicodedata.normalize('NFD', s)
                       if unicodedata.category(c) != 'Mn')

    def decode_tensor_string(tensor_string):
        str_value = ''.join([chr(char) for char in tf.strings.unicode_decode(tensor_string,
                                                                             input_encoding='UTF-8').numpy()])
        return str_value

    def preprocess_sentence(w):
        w = unicode_to_ascii(w.lower().strip())
        w = re.sub(r"([?.!,¿])", r" \1 ", w)
        w = re.sub(r'[" "]+', " ", w)
        w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)
        w = w.strip()
        w = '<start> ' + w + ' <end>'
        return w

    def create_dataset(ds, num_examples):
        pt_pairs = []
        en_pairs = []
        total_pairs = len(ds)
        for x, y in ds.take(int(total_pairs / 10)):
            pt_pairs.append(preprocess_sentence(decode_tensor_string(x)))
            en_pairs.append(preprocess_sentence(decode_tensor_string(y)))

        return pt_pairs, en_pairs

    pt, en = create_dataset(train_ds, None)
    print(en[0])
    print(pt[0])
    print(len(en), len(pt))

    tensor, lang_tokenizer = tokenize(en)
    print(type(tensor))
    print(type(lang_tokenizer))
    ample_string = 'Transformer is awesome.'
    # print("WOrd count", lang_tokenizer.word_counts)
    # print("WOrd index", lang_tokenizer.word_index)

    for ts in lang_tokenizer.word_index:
        print(ts, " - ")

    # print(type(tensor), type(lang_tokenizer))

    # tokenizer_en = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
    #     (en.numpy() for pt, en in train_ds), target_vocab_size=2 ** 13)
    #
    # tokenizer_pt = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
    #     (pt.numpy() for pt, en in train_ds), target_vocab_size=2 ** 13)

    # print(type(tokenizer_en))


if __name__ == "__main__":
    check_version_proxy_gpu()
    run_transformers()
