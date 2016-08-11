from csxnet.data import Sequence
from csxnet.utilities.pure import roots

vocabulary_size = 8000
unknown_token = "<UNK>"
sentence_start_token = "<START>"
sentence_end_token = "<END>"

datapath = roots["csv"] + "reddit.csv"

crossval = 0.2
pca = 0

time = 7
neurons = 4
eta = 0.3


def getrnn(data: Sequence):
    from keras.models import Sequential
    from keras.layers.core import Dense
    from keras.layers.recurrent import LSTM

    net = Sequential()
    net.add(LSTM(input_dim=data.neurons_required()[0], output_dim=30))
    net.add(Dense(output_dim=5))
    net.compile("rmsprop", "mse")

    return net


# def pull_reddit_data(path):
#
#     # Read the data and append SENTENCE_START and SENTENCE_END tokens
#     print("Reading CSV file...")
#     with open(path, 'r') as f:
#         reader = csv.reader(f, skipinitialspace=True)
#         next(reader)
#         # Split full comments into sentences
#         sentences = itertools.chain(*[nltk.sent_tokenize(x[0].lower()) for x in reader])
#         # Append SENTENCE_START and SENTENCE_END
#         sentences = ["%s %s %s" % (sentence_start_token, x, sentence_end_token) for x in sentences]
#     print("Parsed %d sentences." % (len(sentences)))
#
#     # Tokenize the sentences into words
#     # tokenized_sentences = [nltk.word_tokenize(sent) for sent in sentences]
#     #
#     # # Count the word frequencies
#     # word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))
#     # print("Found %d unique words tokens." % len(word_freq.items()))
#     #
#     # # Get the most common words and build index_to_word and word_to_index vectors
#     # vocab = word_freq.most_common(vocabulary_size - 1)
#     # index_to_word = [x[0] for x in vocab]
#     # index_to_word.append(unknown_token)
#     # word_to_index = dict([(w, i) for i, w in enumerate(index_to_word)])
#     #
#     # print("Using vocabulary size %d." % vocabulary_size)
#     # print("The least frequent word in our vocabulary is '%s' and appeared %d times."
#     #       % (vocab[-1][0], vocab[-1][1]))
#     #
#     # # Replace all words not in our vocabulary with the unknown token
#     # for i, sent in enumerate(tokenized_sentences):
#     #     tokenized_sentences[i] = [w if w in word_to_index else unknown_token for w in sent]
#     #
#     # print("\nExample sentence: '%s'" % sentences[0])
#     # print("\nExample sentence after Pre-processing: '%s'" % tokenized_sentences[0])
#     #
#     # # Create the training data
#     # X = np.array([np.array([word_to_index[w] for w in sent[:-1]]) for sent in tokenized_sentences])
#
#     data = Sequence(sentences, embed=True, embeddim=5)
#
#     return data


def pull_reddit_data2(path):
    with open(path, "r") as f:
        lines = f.read()
        lines = lines.split("\n")[1:]
        f.close()

    lines = ["<START> {} <END>".format(line) for line in lines
             if (line not in ("", " ", "\n", "\n\n", "\n\n\n")) and len(line) > 10]
    lines = [line.split(" ") for line in lines]
    lines = [[word for word in line if word] for line in lines]
    raw = []
    for line in lines:
        raw += line
    print("We have {} sentences!".format(len(lines)))

    data = Sequence(raw)
    data.embed(5)
    return data


def main():
    data = pull_reddit_data2(datapath)
    net = getrnn(data)
    print("Teaching network...")
    X, y = data.table()
    net.fit(X, y)

if __name__ == '__main__':
    main()
