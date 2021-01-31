from to_import import *
import unicodedata
from to_encoder import *
from to_attention import *
from to_decoder import *

def run_attention():
    path_to_zip = tf.keras.utils.get_file(
        'spa-eng.zip',
        origin='http://storage.googleapis.com/download.tensorflow.org/data/spa-eng.zip',
        extract=True)

    path_to_file = os.path.dirname(path_to_zip) + "/spa-eng/spa.txt"

    def unicode_to_ascii(word):
        return ''.join(c for c in unicodedata.normalize('NFD', word)
                       if unicodedata.category(c) != 'Mn')

    def decode_tensor_string(tensor_string):
        str_value = ''.join([chr(char) for char in tf.strings.unicode_decode(tensor_string,
                                                        input_encoding='UTF-8').numpy()])
        return str_value

    def preprocess_sentence(w):
        # w = unicode_to_ascii(w.lower().strip())
        w = decode_tensor_string(w.lower().strip())
        w = re.sub(r"([?.!,¿])", r" \1 ", w)
        w = re.sub(r'[" "]+', " ", w)
        w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)
        w = w.strip()
        w = '<start> ' + w + ' <end>'
        return w

    en_sentence = u"May I borrow this book?"
    sp_sentence = u"¿Puedo tomar prestado este libro?"
    #print(preprocess_sentence(en_sentence))
    #print(preprocess_sentence(sp_sentence).encode('utf-8'))

    def tokenize(lang):
        lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
        lang_tokenizer.fit_on_texts(lang)
        tensor = lang_tokenizer.texts_to_sequences(lang)
        tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor,
                                                               padding='post')
        return tensor, lang_tokenizer

    def create_dataset(path, num_examples):
        lines = io.open(path, encoding='UTF-8').read().strip().split('\n')
        word_pairs = [[preprocess_sentence(w) for w in l.split('\t')] for l in lines[:num_examples]]
        return zip(*word_pairs)

    def load_dataset(path, num_examples=None):
        targ_lang, inp_lang = create_dataset(path, num_examples)
        input_tensor, inp_lang_tokenizer = tokenize(inp_lang)
        target_tensor, targ_lang_tokenizer = tokenize(targ_lang)
        return input_tensor, target_tensor, inp_lang_tokenizer, targ_lang_tokenizer

    # en, sp = create_dataset(path_to_file, None)
    # print(en[-1])
    # print(sp[-1])

    NUM_EXAMPLES = 3000
    input_tensor, target_tensor, inp_lang, targ_lang = load_dataset(path_to_file, NUM_EXAMPLES)
    max_length_targ, max_length_inp = target_tensor.shape[1], input_tensor.shape[1]

    input_tensor_train, \
    input_tensor_val, \
    target_tensor_train, \
    target_tensor_val = train_test_split(input_tensor, target_tensor, test_size=0.2)

    # print("@"*50)
    # print(input_tensor_train[0])
    # print(target_tensor_train[0])
    # print(input_tensor_val[0])
    # print(target_tensor_val[0])

    # print(len(input_tensor_train), type(input_tensor_train),
    #       len(target_tensor_train),
    #       len(input_tensor_val), type(input_tensor_val),
    #       len(target_tensor_val))

    # def convert(lang, tensor):
    #     for t in tensor:
    #         if t != 0:
    #             print("%d ----> %s" % (t, lang.index_word[t]))
    #
    # print("Input Language; index to word mapping")
    # convert(inp_lang, input_tensor_train[0])
    # print()
    # print("Target Language; index to word mapping")
    # convert(targ_lang, target_tensor_train[0])

    BUFFER_SIZE = len(input_tensor_train)
    BATCH_SIZE = 64
    steps_per_epoch = len(input_tensor_train) // BATCH_SIZE
    embedding_dim = 256
    UNITS = 1024
    vocab_inp_size = len(inp_lang.word_index) + 1
    vocab_tar_size = len(targ_lang.word_index) + 1

    tensor_train = (input_tensor_train, target_tensor_train)
    dataset = tf.data.Dataset.from_tensor_slices(tensor_train)
    dataset = dataset.shuffle(BUFFER_SIZE)
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)

    # example_input_batch, example_target_batch = next(iter(dataset))
    # print(example_input_batch.shape, example_target_batch.shape)
    # print(type(example_input_batch), type(example_target_batch.shape))


    # ENCODER
    encoder = EncoderBahdanau(vocab_inp_size, embedding_dim, UNITS, BATCH_SIZE)

    # ATTENTION LAYER
    attention_layer = BahdanauAttention(10)

    # DECODER
    decoder = Decoder(vocab_tar_size, embedding_dim, UNITS, BATCH_SIZE)

    # OPTIMIZER
    optimizer = tf.keras.optimizers.Adam()
    # LOSS
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,
                                                                reduction='none')

    def loss_function(real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = loss_object(real, pred)

        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask

        return tf.reduce_mean(loss_)

    cp_dir = "./checkpoints"
    cp_prefix = os.path.join(cp_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                     encoder=encoder,
                                     decoder=decoder)

    @tf.function
    def train_step(inp, targ, enc_hidden):
        loss = 0

        with tf.GradientTape() as tape:
            enc_output, enc_hidden = encoder(inp, enc_hidden)
            dec_hidden = enc_hidden
            dec_input = tf.expand_dims([targ_lang.word_index['<start>']] * BATCH_SIZE, 1)

            for t in range(1, targ.shape[1]):
                predictions, dec_input, _ = decoder(dec_input, dec_hidden, enc_output)

                loss += loss_function(targ[:, t], predictions)
                dec_input = tf.expand_dims(targ[:, t], 1)

        batch_loss = (loss / int(targ.shape[1]))
        variables = encoder.trainable_variables + decoder.trainable_variables
        gradients = tape.gradient(loss, variables)
        optimizer.apply_gradients(zip(gradients, variables))

        return batch_loss

    EPOCHS = 10

    for epoch in range(EPOCHS):
        start = time.time()
        enc_hidden = encoder.initialize_hidden_state()
        total_loss = 0

        for(batch, (inp, targ)) in enumerate(dataset.take(steps_per_epoch)):
            batch_loss = train_step(inp, targ, enc_hidden)
            total_loss += batch_loss

            if batch % 100 == 0:
                print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                             batch,
                                                             batch_loss.numpy()))
            if (epoch + 1) % 2 == 0:
                checkpoint.save(file_prefix=cp_prefix)

            print('Epoch {} Loss {:.4f}'.format(epoch + 1,
                                                total_loss / steps_per_epoch))
            print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))





if __name__ == "__main__":
    check_version_proxy_gpu()
    run_attention()
