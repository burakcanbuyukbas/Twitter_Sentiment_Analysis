import numpy as np
import sklearn as sklearn
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding, Flatten
from keras.layers import Conv1D
from keras.callbacks import ReduceLROnPlateau
import utils
from keras.preprocessing.sequence import pad_sequences
import re
from nltk import FreqDist
import pickle
from utils import write_status
from collections import Counter
import pandas as pd


def analyze_tweet(tweet):
    result = {}
    result['MENTIONS'] = tweet.count('USER_MENTION')
    result['URLS'] = tweet.count('URL')
    result['POS_EMOS'] = tweet.count('EMO_POS')
    result['NEG_EMOS'] = tweet.count('EMO_NEG')
    tweet = tweet.replace('USER_MENTION', '').replace(
        'URL', '')
    words = tweet.split()
    result['WORDS'] = len(words)
    bigrams = get_bigrams(words)
    result['BIGRAMS'] = len(bigrams)
    return result, words, bigrams


def get_bigrams(tweet_words):
    bigrams = []
    num_words = len(tweet_words)
    for i in range(num_words - 1):
        bigrams.append((tweet_words[i], tweet_words[i + 1]))
    return bigrams


def get_bigram_freqdist(bigrams):
    freq_dict = {}
    for bigram in bigrams:
        if freq_dict.get(bigram):
            freq_dict[bigram] += 1
        else:
            freq_dict[bigram] = 1
    counter = Counter(freq_dict)
    return counter


def preprocess_word(word):
    # Remove punctuation
    word = word.strip('\'"?!,.():;')
    # Convert more than 2 letter repetitions to 2 letter
    # funnnnny --> funny
    word = re.sub(r'(.)\1+', r'\1\1', word)
    # Remove - & '
    word = re.sub(r'(-|\')', '', word)
    return word


def is_valid_word(word):
    # Check if word begins with an alphabet
    return (re.search(r'^[a-zA-Z][a-z0-9A-Z\._]*$', word) is not None)


def handle_emojis(tweet):
    # Smile -- :), : ), :-), (:, ( :, (-:, :')
    tweet = re.sub(r'(:\s?\)|:-\)|\(\s?:|\(-:|:\'\))', ' EMO_POS ', tweet)
    # Laugh -- :D, : D, :-D, xD, x-D, XD, X-D
    tweet = re.sub(r'(:\s?D|:-D|x-?D|X-?D)', ' EMO_POS ', tweet)
    # Love -- <3, :*
    tweet = re.sub(r'(<3|:\*)', ' EMO_POS ', tweet)
    # Wink -- ;-), ;), ;-D, ;D, (;,  (-;
    tweet = re.sub(r'(;-?\)|;-?D|\(-?;)', ' EMO_POS ', tweet)
    # Sad -- :-(, : (, :(, ):, )-:
    tweet = re.sub(r'(:\s?\(|:-\(|\)\s?:|\)-:)', ' EMO_NEG ', tweet)
    # Cry -- :,(, :'(, :"(
    tweet = re.sub(r'(:,\(|:\'\(|:"\()', ' EMO_NEG ', tweet)
    return tweet


def preprocess_tweet(tweet):
    processed_tweet = []
    # Convert to lower case
    tweet = tweet.lower()
    # Replaces URLs with the word URL
    tweet = re.sub(r'((www\.[\S]+)|(https?://[\S]+))', ' URL ', tweet)
    # Replace @handle with the word USER_MENTION
    tweet = re.sub(r'@[\S]+', 'USER_MENTION', tweet)
    # Replaces #hashtag with hashtag
    tweet = re.sub(r'#(\S+)', r' \1 ', tweet)
    # Remove RT (retweet)
    tweet = re.sub(r'\brt\b', '', tweet)
    # Replace 2+ dots with space
    tweet = re.sub(r'\.{2,}', ' ', tweet)
    # Strip space, " and ' from tweet
    tweet = tweet.strip(' "\'')
    # Replace emojis with either EMO_POS or EMO_NEG
    tweet = handle_emojis(tweet)
    # Replace multiple spaces with a single space
    tweet = re.sub(r'\s+', ' ', tweet)
    words = tweet.split()

    for word in words:
        word = preprocess_word(word)
        if is_valid_word(word):
            processed_tweet.append(word)

    return ' '.join(processed_tweet)


def preprocess_csv(csv_file_name, processed_file_name, test_file):
    save_to_file = open(processed_file_name, 'w')
    df = pd.read_csv(csv_file_name, sep="\t")
    total = len(df.index)
    for i, line in df.iterrows():
        if not test_file:
            tweet = line[1]
            sentiment = int(line[0])
        else:
            tweet = line[0]
        processed_tweet = preprocess_tweet(tweet)
        if not test_file:
            #save_to_file.write('%d,%s\n' %(sentiment, processed_tweet))
            save_to_file.write(str(sentiment) + ',' + processed_tweet + '\n')
        else:
            save_to_file.write('%s\n' %
                               (processed_tweet))
        write_status(i + 1, total)


    save_to_file.close()
    print('\nSaved processed tweets to: %s' % processed_file_name)
    return processed_file_name


def get_glove_vectors(vocab):
    """
    Extracts glove vectors from seed file only for words present in vocab.
    """
    print('Looking for GLOVE seeds')
    glove_vectors = {}
    found = 0
    with open(GLOVE_FILE, 'r', encoding="utf8") as glove_file:
        for i, line in enumerate(glove_file):
            utils.write_status(i + 1, 1193514)
            tokens = line.strip().split()
            word = tokens[0]
            if vocab.get(word):
                vector = [float(e) for e in tokens[1:]]
                glove_vectors[word] = np.array(vector)
                found += 1
    print('\n')
    return glove_vectors


def get_feature_vector(tweet):
    """
    Generates a feature vector for each tweet where each word is
    represented by integer index based on rank in vocabulary.
    """
    words = (str(tweet)).split()
    feature_vector = []
    for i in range(len(words) - 1):
        word = words[i]
        if vocab.get(word) is not None:
            feature_vector.append(vocab.get(word))
    if len(words) >= 1:
        if vocab.get(words[-1]) is not None:
            feature_vector.append(vocab.get(words[-1]))
    return feature_vector


def process_tweets(csv_file, test_file=False):
    """
    Generates training X, y pairs.
    """
    tweets = []
    labels = []
    print('Generating feature vectors')
    with open(csv_file, 'r') as csv:
        lines = csv.readlines()
        total = len(lines)
        for i, line in enumerate(lines):
            if test_file:
                tweet = line
            else:
                sentiment, tweet = line.split(',')
            feature_vector = get_feature_vector(tweet)
            if test_file:
                tweets.append(feature_vector)
            else:
                tweets.append(feature_vector)
                labels.append(int(sentiment))
            utils.write_status(i + 1, total)
    print('\n')
    return tweets, np.array(labels)



FREQ_DIST_FILE = 'data/train-processed-freqdist.pkl'
BI_FREQ_DIST_FILE = 'data/train-processed-freqdist-bi.pkl'
TRAIN_PROCESSED_FILE = 'data/train-processed.csv'
TEST_PROCESSED_FILE = 'data/test-processed.csv'
GLOVE_FILE = 'glove.twitter.27B.200d.txt'
csv_file_name = 'data/train.tsv'
processed_file_name = 'data/train-processed.csv'
np.random.seed(1337)
dim = 200
vocab_size = 90000
batch_size = 500
max_length = 40
kernel_size = 3
vocab = utils.top_n_words(FREQ_DIST_FILE, vocab_size, shift=1)

train = 0
#train = 'train-processed.csv'

#preprocess_csv('dev-0/in.tsv', 'dev-0/test-processed.csv', test_file=True)


if train:


    preprocess_csv('data/train.tsv', 'data/train-processed.csv', test_file=False)
    preprocess_csv('data/in.tsv', 'data/test-processed.csv', test_file=True)

    num_tweets, num_pos_tweets, num_neg_tweets = 0, 0, 0
    num_mentions, max_mentions = 0, 0
    num_emojis, num_pos_emojis, num_neg_emojis, max_emojis = 0, 0, 0, 0
    num_urls, max_urls = 0, 0
    num_words, num_unique_words, min_words, max_words = 0, 0, 1e6, 0
    num_bigrams, num_unique_bigrams = 0, 0
    all_words = []
    all_bigrams = []
    with open('data/train-processed.csv', 'r') as csv:
        lines = csv.readlines()
        num_tweets = len(lines)
        for i, line in enumerate(lines):
            if_pos, tweet = line.strip().split(',')
            if_pos = int(if_pos)
            if if_pos:
                num_pos_tweets += 1
            else:
                num_neg_tweets += 1
            result, words, bigrams = analyze_tweet(tweet)
            num_mentions += result['MENTIONS']
            max_mentions = max(max_mentions, result['MENTIONS'])
            num_pos_emojis += result['POS_EMOS']
            num_neg_emojis += result['NEG_EMOS']
            max_emojis = max(
                max_emojis, result['POS_EMOS'] + result['NEG_EMOS'])
            num_urls += result['URLS']
            max_urls = max(max_urls, result['URLS'])
            num_words += result['WORDS']
            min_words = min(min_words, result['WORDS'])
            max_words = max(max_words, result['WORDS'])
            all_words.extend(words)
            num_bigrams += result['BIGRAMS']
            all_bigrams.extend(bigrams)
            write_status(i + 1, num_tweets)
    num_emojis = num_pos_emojis + num_neg_emojis
    unique_words = list(set(all_words))
    with open('data/train-processed-unique.txt', 'w') as uwf:
        uwf.write('\n'.join(unique_words))
    num_unique_words = len(unique_words)
    num_unique_bigrams = len(set(all_bigrams))
    print('\nCalculating frequency distribution')
    # Unigrams
    freq_dist = FreqDist(all_words)
    pkl_file_name = 'data/train-processed-freqdist.pkl'
    with open(pkl_file_name, 'wb') as pkl_file:
        pickle.dump(freq_dist, pkl_file)
    print('Saved uni-frequency distribution to %s' % pkl_file_name)
    # Bigrams
    bigram_freq_dist = get_bigram_freqdist(all_bigrams)
    bi_pkl_file_name = 'data/train-processed-freqdist-bi.pkl'
    with open(bi_pkl_file_name, 'wb') as pkl_file:
        pickle.dump(bigram_freq_dist, pkl_file)
    print('Saved bi-frequency distribution to %s' % bi_pkl_file_name)
    print('\n[Analysis Statistics]')
    print('Tweets => Total: %d, Positive: %d, Negative: %d' % (num_tweets, num_pos_tweets, num_neg_tweets))
    print('User Mentions => Total: %d, Avg: %.4f, Max: %d' % (
    num_mentions, num_mentions / float(num_tweets), max_mentions))
    print('URLs => Total: %d, Avg: %.4f, Max: %d' % (num_urls, num_urls / float(num_tweets), max_urls))
    print('Emojis => Total: %d, Positive: %d, Negative: %d, Avg: %.4f, Max: %d' % (
    num_emojis, num_pos_emojis, num_neg_emojis, num_emojis / float(num_tweets), max_emojis))
    print('Words => Total: %d, Unique: %d, Avg: %.4f, Max: %d, Min: %d' % (
    num_words, num_unique_words, num_words / float(num_tweets), max_words, min_words))
    print('Bigrams => Total: %d, Unique: %d, Avg: %.4f' % (
    num_bigrams, num_unique_bigrams, num_bigrams / float(num_tweets)))


    glove_vectors = get_glove_vectors(vocab)
    tweets, labels = process_tweets(TRAIN_PROCESSED_FILE, test_file=False)
    # Create and embedding matrix
    embedding_matrix = np.random.randn(vocab_size + 1, dim) * 0.01
    # Seed it with GloVe vectors
    for word, i in vocab.items():
        glove_vector = glove_vectors.get(word)
        if glove_vector is not None:
            embedding_matrix[i] = glove_vector
    tweets = pad_sequences(tweets, maxlen=max_length, padding='post')
    # pad_sequences is used to ensure that all sequences in a list have the same length).
    shuffled_indices = np.random.permutation(tweets.shape[0])
    # Shuffling data serves the purpose of reducing variance and making sure that models remain general
    # and overfit less.
    tweets = tweets[shuffled_indices]
    labels = labels[shuffled_indices]



    
    model = Sequential()
    # Word embeddings provide a dense representation of words and their relative meanings.
    model.add(Embedding(vocab_size + 1, dim, weights=[embedding_matrix], input_length=max_length))

    # dropout refers to ignoring units (i.e. neurons) during
    # the training phase of certain set of neurons which is chosen at random to prevent over-fitting”.
    # A fully connected layer occupies most of the parameters, and neurons develop co-dependency amongst
    # each other during training which holds the individual power of each neuron leading to over-fitting.
    # Dropout forces a neural network to learn more robust features by ignoring some individual nodes.
    model.add(Dropout(0.4))

    model.add(Conv1D(600, kernel_size, padding='valid', activation='relu', strides=1))
    model.add(Conv1D(300, kernel_size, padding='valid', activation='relu', strides=1))
    model.add(Conv1D(150, kernel_size, padding='valid', activation='relu', strides=1))
    model.add(Conv1D(75, kernel_size, padding='valid', activation='relu', strides=1))
    # filters: the dimensionality of the output space (i.e. the number of output filters in the convolution).
    # kernel_size: An integer that specifying the length of the 1D convolution window.
    # valid padding means no padding
    # activation : Rectified Linear Unit.(RELU) With default values, it returns element-wise max(x, 0).
    # strides: An integer or tuple/list of a single integer, specifying the stride length of the convolution. Specifying any stride value != 1 is incompatible with specifying any dilation_rate value != 1.

    model.add(Flatten())
    # flattening pooled feature map into a column

    model.add(Dense(600))
    # A dense layer is just a regular layer of neurons in a neural network.
    # Each neuron recieves input from all the neurons in the previous layer.
    # The layer has a weight matrix W, bias vector b, and the activations of previous layer a.

    model.add(Dropout(0.5))
    # Dropout again

    model.add(Activation('relu'))
    # Activation Layer is an activation function that decides the final value of a neuron.

    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # For a binary classification like our example, the typical loss function is the binary cross-entropy / log loss.
    # Adam(Adaptive Moment Estimation) is a method that computes adaptive learning rates for each parameter and
    # updates the model parameters such as Weights and Bias values

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=0.000001)
    # ReduceLROnPlateau will adjust the learning rate when a plateau in model performance is detected,
    # e.g. no change for a given number of training epochs.
    # This callback is designed to reduce the learning rate after the model stops improving
    # with the hope of fine-tuning model weights.

    model.fit(tweets, labels, batch_size=128, epochs=1, validation_split=0.1, shuffle=True)
    # Batch size: the number of samples that will be propagated through the network.

    model.save("model1.h5")
    ######################



    ######################
else:
    model = load_model("model2.h5")
    print(model.summary())
    test_tweets, _ = process_tweets('dev-0/test-processed.csv', test_file=True) # TEST_PROCESSED_FILE
    test_tweets = pad_sequences(test_tweets, maxlen=40, padding='post')
    predictions = model.predict(test_tweets, batch_size=128, verbose=1, )
    results = zip(map(str, range(len(test_tweets))), np.round(predictions[:, 0]).astype(int))
    utils.save_results_to_csv(results, 'data/cnn.csv')

    predicted = pd.read_csv('data/cnn.csv').values
    expected = pd.read_csv('dev-0/expected.tsv', sep="\t").values[1:]
    # score = model.evaluate(predicted.values, expected.values, verbose=1)
    acc = sklearn.metrics.accuracy_score(predicted, expected)
    print('Test accuracy:', acc)
