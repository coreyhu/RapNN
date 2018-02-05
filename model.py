import pronouncing
import re
import random
import os
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import LSTM, Dense, Dropout
from tensorflow.python.keras.optimizers import RMSprop
import markovify
import numpy as np
import pandas as pd


max_syllables = 16
train_file = "lyrics-songs.txt"
name = "rapper"
save_location = "rap.txt"
train_mode = True
new_lyric_data = False


def create_model():
    model = Sequential()
    model.add(LSTM(16, input_shape=(2, 2), return_sequences=True, activation='tanh'))
    model.add(LSTM(32, return_sequences=True, activation='tanh'))
    model.add(LSTM(64, return_sequences=True, activation='tanh'))
    model.add(LSTM(128, return_sequences=True, activation='tanh'))
    model.add(LSTM(64, return_sequences=True, activation='tanh'))
    model.add(Dropout(.15))
    model.add(LSTM(32, return_sequences=True, activation='tanh'))
    model.add(Dense(16, activation='sigmoid'))
    model.add(LSTM(8, return_sequences=True, activation='relu'))
    model.add(Dense(2, activation='softmax'))
    model.summary()

    rms = RMSprop(lr=.01)
    model.compile(optimizer=rms, loss='categorical_crossentropy', metrics=['accuracy'])

    if "weights.h5" in os.listdir(".") and train_mode == False:
        model.load_weights(str("weights.h5"))
        print("loading saved network: weights.h5")
    return model


def markov():
    read = open(train_file, 'r', encoding="utf8").read()
    text_model = markovify.NewlineText(read)
    return text_model


def markov_pandas(column):
    text = column.str.cat(sep='\n ')
    text_model = markovify.NewlineText(text)
    return text_model


def syllables(line):
    count = 0
    split_line = line.split(" ")
    for word in split_line:
        vowels = 'aeiouy'
        word = word.lower().strip(".:;?!")
        if len(word) == 0:
            continue
        if word[0] in vowels:
            count += 1
        for index in range(1, len(word)):
            if word[index] in vowels and word[index - 1] not in vowels:
                count += 1
        if word.endswith('e'):
            count -= 1
        if word.endswith('le'):
            count += 1
        if count == 0:
            count += 1
    return count / max_syllables


def rhymeindex(lyrics):
    if str(name) + ".rhymes" in os.listdir(".") and not new_lyric_data:
        print("loading saved rhymes from " + str(name) + ".rhymes")
        rhymes = open(str(name) + ".rhymes", "r").read().split("\n")
        print("done")
        return rhymes
    else:
        rhyme_master_list = []
        print("building rhymes for all lines")
        count = 0
        lyrics = [x for x in lyrics if x != '' or str(x) != 'nan']
        lyrics_len = len(lyrics)
        for i in lyrics:
            word = re.sub(r"\W+", '', i.split(" ")[-1]).lower()
            rhymeslist = pronouncing.rhymes(word)
            rhymeslist_ends = [r[-2:] for r in rhymeslist]

            try:
                rhymescheme = max(set(rhymeslist_ends), key=rhymeslist_ends.count)
            except Exception:
                rhymescheme = word[-2:]

            rhyme_master_list.append(rhymescheme)
            if count % 50 == 0:
                print("building rhyme list: ", count, "/", lyrics_len, " (", "{0:.4f}".format(count / lyrics_len), ")")
            count += 1

        print("building rhyme list: ", count, "/", lyrics_len, " (Done)")
        rhyme_master_list = list(set(rhyme_master_list))

        reverselist = [x[::-1] for x in rhyme_master_list]
        reverselist = sorted(reverselist)

        rhymelist = [x[::-1] for x in reverselist]

        f = open(str(name) + ".rhymes", "w")
        f.write("\n".join(rhymelist))
        f.close()
        print(rhymelist)
        return rhymelist


def rhyme(line, rhyme_list):
    word = re.sub(r"\W+", '', line.split(" ")[-1]).lower()
    rhymeslist = pronouncing.rhymes(word)
    rhymeslistends = [i[-2:] for i in rhymeslist]
    try:
        rhyme_scheme = max(set(rhymeslistends), key=rhymeslistends.count)
    except Exception:
        rhyme_scheme = word[-2:]
    try:
        float_rhyme = rhyme_list.index(rhyme_scheme)
        float_rhyme = float_rhyme / float(len(rhyme_list))
        return float_rhyme
    except ValueError:
        return None


def split_lyrics_file(text_file):
    print("splitting lyrics")
    text = open(text_file, encoding="utf8").read()
    text = text.split("\n")
    while "" in text:
        text.remove("")

    df = pd.DataFrame(text, columns=['lyrics'])
    df.to_csv('bars.csv', index=False)

    print("done")
    return text


def generate_lyrics(markov_model):
    bars = []
    last_words = []
    lyric_length = len(open(train_file, encoding="utf8").read().split("\n"))
    count = 0

    while len(bars) < lyric_length / 9 and count < lyric_length * 2:
        line = markov_model.make_sentence()

        if line is not None and syllables(line) < 1:
            def get_last_word(bar):
                line_last_word = bar.split(" ")[-1]
                if line_last_word[-1] in "!.?,":
                    line_last_word = line_last_word[:-1]
                return line_last_word

            last_word = get_last_word(line)
            if line not in bars and last_words.count(last_word) < 3:
                bars.append(line)
                last_words.append(last_word)
                count += 1
    return bars


def build_dataset(lines, rhyme_list):
    if 'x_data.npy' in os.listdir(".") and 'y_data.npy' in os.listdir(".") and not new_lyric_data:
        print("loading dataset from disk")
        x_data = np.load('x_data.npy')
        y_data = np.load('y_data.npy')
        print("done")

    else:
        print("building dataset")
        dataset = []

        x_data = []
        y_data = []

        line_len = len(lines) - 3
        i = 0

        for i in range(line_len):
            for x in range(4):
                if len(dataset) - 1 < i + x:
                    line = lines[i + x]
                    line_list = [line, syllables(line), rhyme(line, rhyme_list)]
                    dataset.append(line_list)

            line1 = dataset[i][1:]
            line2 = dataset[i + 1][1:]
            line3 = dataset[i + 2][1:]
            line4 = dataset[i + 3][1:]

            x = [line1[0], line1[1], line2[0], line2[1]]
            x = np.array(x)
            x = x.reshape(2, 2)

            y = [line3[0], line3[1], line4[0], line4[1]]
            y = np.array(y)
            y = y.reshape(2, 2)

            if True in pd.isnull(x) or True in pd.isnull(y):
                continue

            x_data.append(x)
            y_data.append(y)

            if i % 50 == 0:
                print("building dataset: ", i, "/", line_len, " (", "{0:.4f}".format(i / line_len), ")")

        print("building dataset: ", i, "/", line_len, " (Done)")
        x_data = np.array(x_data)
        y_data = np.array(y_data)

        np.save("x_data", x_data)
        np.save("y_data", y_data)

    print("done")
    return x_data, y_data


def compose_rap(rhyme_list, model):
    rap_vectors = []

    df = pd.read_csv('bars.csv', names=['lyrics'], header=None)
    human_lyrics = df.lyrics.tolist()[1:]

    initial_index = random.choice(range(len(human_lyrics) - 1))
    initial_lines = human_lyrics[initial_index:initial_index + 2]

    starting_input = []
    for line in initial_lines:
        starting_input.append([syllables(line), rhyme(line, rhyme_list)])

    starting_vectors = model.predict(np.array([starting_input]).flatten().reshape(1, 2, 2))
    rap_vectors.append(starting_vectors)

    for i in range(100):
        rap_vectors.append(model.predict(np.array([rap_vectors[-1]]).flatten().reshape(1, 2, 2)))

    return rap_vectors


def vectors_into_song(vectors, generated_lyrics, rhyme_list):
    print("\n\n")
    print("About to write song (this could take a moment)...")
    print("\n\n")

    def last_word_compare(rap, line2):
        cost = 0
        for line1 in rap:
            word1 = line1.split(" ")[-1]
            word2 = line2.split(" ")[-1]

            while word1[-1] in "?!,. ":
                word1 = word1[:-1]

            while word2[-1] in "?!,. ":
                word2 = word2[:-1]

            if word1 == word2:
                cost += 0.2

        return cost

    def calculate_score(vector_half, syllables, rhyme, penalty):
        desired_syllables = vector_half[0]
        desired_rhyme = vector_half[1]
        desired_syllables = desired_syllables * max_syllables
        desired_rhyme = desired_rhyme * len(rhyme_list)

        score = 1.0 - (
        abs((float(desired_syllables) - float(syllables))) + abs((float(desired_rhyme) - float(rhyme)))) - penalty

        return score

    dataset = []
    for line in generated_lyrics:
        line_list = [line, syllables(line), rhyme(line, rhyme_list)]
        dataset.append(line_list)

    rap = []
    vector_halves = []

    for vector in vectors:
        vector_halves.append(list(vector[0][0]))
        vector_halves.append(list(vector[0][1]))

    for vector in vector_halves:
        score_list = []
        for item in dataset:
            line = item[0]

            if len(rap) != 0:
                penalty = last_word_compare(rap, line)
            else:
                penalty = 0
            total_score = calculate_score(vector, item[1], item[2], penalty)
            score_entry = [line, total_score]
            score_list.append(score_entry)

        fixed_score_list = map(lambda x: float(x[1]), score_list)
        max_score = max(fixed_score_list)

        for item in score_list:
            if item[1] == max_score:
                rap.append(item[0])
                print(str(item[0]))

                for i in dataset:
                    if item[0] == i[0]:
                        dataset.remove(i)
                        break
                break
    return rap


def train(x_data, y_data, model):
    model.fit(np.array(x_data), np.array(y_data), batch_size=32, epochs=10, )
    model.save_weights(name + ".rap")


def main():
    model = create_model()
    inp = input("Have you updated {}? (Y/N): ".format(train_file))

    if inp.upper() == "Y":
        global new_lyric_data
        new_lyric_data = True

    if train_mode:
        if new_lyric_data:
            bars = split_lyrics_file(train_file)
        else:
            df = pd.read_csv('bars.csv', names=['lyrics'], header=None)
            bars = df.lyrics.tolist()[1:]

        rhyme_list = rhymeindex(bars)
        x_data, y_data = build_dataset(bars, rhyme_list)
        data_size = x_data.shape[0]
        split_index = data_size // 5
        x_train, x_test = x_data[split_index:], x_data[:split_index]
        y_train, y_test = y_data[split_index:], y_data[:split_index]
        train(x_train, y_train, model)

        score = model.evaluate(x_test, y_test, verbose=1)
        print("Test score: ", score[0])
        print("Test accuracy: ", score[1])

    else:
        text_model = markov()
        bars = generate_lyrics(text_model)
        rhyme_list = rhymeindex(bars)
        vectors = compose_rap(rhyme_list, model)
        rap = vectors_into_song(vectors, bars, rhyme_list)
        f = open(name + "-lyrics.txt", "w")
        for bar in rap:
            print(bar)
            f.write(bar)
            f.write("\n")


main()
