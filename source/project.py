import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from pandas import DataFrame
import unicodedata
import re
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import torch.utils.data as Data

# Approach 1: Baseline LR
def preprocess_reviews(reviews):
    modified_reviews =[]
    remove_char = '[’!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~\n。！，]+'
    for review in reviews:
        modified_review = remove_char.replace('<br />', "")
        modified_review = re.sub(modified_review, '', review)
        modified_reviews.append(modified_review)

    return modified_reviews

def input_file(train, test):
    # Input training data
    reviews_train = train
    paths_train = ['database/aclImdb/train/neg', 'database/aclImdb/train/pos']
    for path in paths_train:
        files = os.listdir(path)
        for file in files:
            if not os.path.isdir(file):
                temp = file.split("_", 1)
                score = temp[1].split(".", 1)
                review = ""
                with open(path + "/" + file, 'r', encoding="utf8") as f:
                    for line in f.readlines():
                        review += line.strip()
                # reviews_train.append(' '.join([str(score[0]), review]))
                reviews_train.append(review)

    # Input training data
    reviews_test = test
    paths_test = ['database/aclImdb/test/neg', 'database/aclImdb/test/pos']
    for path in paths_test:
        files = os.listdir(path)
        for file in files:
            if not os.path.isdir(file):
                temp = file.split("_", 1)
                score = temp[1].split(".", 1)
                review = ''
                with open(path + "/" + file, 'r', encoding="utf8") as f:
                    for line in f.readlines():
                        review += line.strip()
                # reviews_test.append(' '.join([str(score[0]), review]))
                reviews_test.append(review)

    result = preprocess_reviews(train) + preprocess_reviews(test)
    return result

def convert_to_pickle(item, directory):
    pickle.dump(item, open(directory, "wb"))


def load_from_pickle(directory):
    return pickle.load(open(directory, "rb"))

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

def normalizeString(s):
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    s = unicodeToAscii(s.lower().strip())
    return s

def epoch_time(start, end):
    time = end - start
    mins = int(time / 60)
    secs = int(time - (mins * 60))

    return mins, secs

class Lang:
    def __init__(self):
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words = self.n_words + 1
        else:
            self.word2count[word] = self.word2count[word] + 1

def convertWord2index(word):
    if lang_process.word2index.get(word) == None:
        return 1
    else:
        return lang_process.word2index.get(word)

def pad_sequences(x, max_len):
    padded = np.zeros((max_len), dtype=np.int64)
    if len(x) > max_len:
        padded[:] = x[:max_len]
    else:
        padded[:len(x)] = x
    return padded

def accuracy(target, logit):
    """ Obtain accuracy for training round """
    target = torch.max(target, 1)[1]  # convert from one-hot encoding to class indices
    corrects = (logit == target).sum()
    accuracy = 100.0 * corrects / len(logit)
    return accuracy

class EmotionRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_units, batch_sz, output_size):
        super(EmotionRNN, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_units = hidden_units
        self.batch_sz = batch_sz
        self.output_size = output_size

        # layers
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.rnn = nn.RNN(self.embedding_dim, self.hidden_units, batch_first=True)
        self.fc = nn.Linear(self.hidden_units, self.output_size)

    def forward(self, x):
        self.batch_sz = x.size(0)
        x = self.embedding(x)
        output, hidden = self.rnn(x)
        assert torch.equal(output[:, -1, :], hidden.squeeze(0))
        output = self.fc(output[:, -1, :])
        output = F.log_softmax(output, dim=1)
        return output

class EmotionGRU(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_units, batch_sz, output_size, layers=2):
        super(EmotionGRU, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_units = hidden_units
        self.batch_sz = batch_sz
        self.output_size = output_size
        self.num_layers = layers

        # layers
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.dropout = nn.Dropout(0.75)
        self.gru = nn.GRU(self.embedding_dim, self.hidden_units, num_layers=self.num_layers, batch_first=True,
                          bidirectional=False, dropout=0.75)
        self.fc = nn.Linear(self.hidden_units, self.output_size)

    def init_hidden(self):
        return torch.zeros((self.num_layers, self.batch_sz, self.hidden_units)).to(device)

    def forward(self, x):
        self.batch_sz = x.size(0)
        x = self.embedding(x)
        self.hidden = self.init_hidden()
        output, self.hidden = self.gru(x, self.hidden)
        output = output[:, -1, :]
        output = self.dropout(output)
        output = self.fc(output)
        return output

class Bi_EmotionGRU(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_units, batch_sz, output_size, layers=2):
        super(Bi_EmotionGRU, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_units = hidden_units
        self.batch_sz = batch_sz
        self.output_size = output_size
        self.num_layers = layers

        # layers
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.dropout = nn.Dropout(0.5)
        self.gru = nn.GRU(self.embedding_dim, self.hidden_units, num_layers=self.num_layers, batch_first=True,
                          bidirectional=True, dropout=0.5)
        self.fc = nn.Linear(self.hidden_units * 2, self.output_size)

    def init_hidden(self):
        return torch.zeros((self.num_layers * 2, self.batch_sz, self.hidden_units)).to(device)

    def forward(self, x):
        self.batch_sz = x.size(0)
        x = self.embedding(x)
        self.hidden = self.init_hidden()
        output, self.hidden = self.gru(x, self.hidden)
        output = output[:, -1, :]
        output = self.dropout(output)
        output = self.fc(output)
        return output

class EmotionLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_units, batch_sz, output_size, layers=2):
        super(EmotionLSTM, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_units = hidden_units
        self.batch_sz = batch_sz
        self.output_size = output_size
        self.num_layers = layers

        # layers
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.dropout = nn.Dropout(0.75)
        self.gru = nn.LSTM(self.embedding_dim, self.hidden_units, num_layers=self.num_layers, batch_first=True,
                           bidirectional=False, dropout=0.75)
        self.fc = nn.Linear(self.hidden_units, self.output_size)

    def init_hidden(self):
        return (torch.zeros((self.num_layers, self.batch_sz, self.hidden_units)).to(device),
                torch.zeros((self.num_layers, self.batch_sz, self.hidden_units)).to(device))

    def forward(self, x):
        self.batch_sz = x.size(0)
        x = self.embedding(x)
        (self.hidden, self.cell_state) = self.init_hidden()
        output, (self.hidden, self.cell_state) = self.gru(x, (self.hidden, self.cell_state))

        output = output[:, -1, :]
        output = self.dropout(output)
        output = self.fc(output)
        return output

class Bi_EmotionLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_units, batch_sz, output_size, layers=2):
        super(Bi_EmotionLSTM, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_units = hidden_units
        self.batch_sz = batch_sz
        self.output_size = output_size
        self.num_layers = layers

        # layers
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.dropout = nn.Dropout(0.5)
        self.gru = nn.LSTM(self.embedding_dim, self.hidden_units, num_layers=self.num_layers, batch_first=True,
                           bidirectional=True, dropout=0.5)
        self.fc = nn.Linear(self.hidden_units * 2, self.output_size)

    def init_hidden(self):
        return (torch.zeros((self.num_layers * 2, self.batch_sz, self.hidden_units)).to(device),
                torch.zeros((self.num_layers * 2, self.batch_sz, self.hidden_units)).to(device))

    def forward(self, x):
        self.batch_sz = x.size(0)
        x = self.embedding(x)
        (self.hidden, self.cell_state) = self.init_hidden()
        output, (self.hidden, self.cell_state) = self.gru(x, (self.hidden, self.cell_state))

        output = output[:, -1, :]
        output = self.dropout(output)
        output = self.fc(output)
        return output

if __name__ == '__main__':
    # Loading data and preprocess it
    reviews_train = []
    reviews_test = []
    result = input_file(reviews_train, reviews_test)

    # Save it to a single csv file
    target = ['Negative' if i < 12500 or (25000 <= i < 37500) else 'Positive' for i in range(50000)]
    index = [(i + 1) for i in range(50000)]

    train = {}
    train['ID'] = index
    train['review'] = result
    train['label'] = target
    df = DataFrame(train)
    df.to_csv('allData.csv', index=False)

    reviews_train_clean = result[:25000]
    reviews_test_clean = result[25000:]


    # Baseline (one-hot vector + Logistic Regression)
    print("====================Logistic Regression====================")
    cv = CountVectorizer(binary=True)
    cv.fit(reviews_train_clean)
    X = cv.transform(reviews_train_clean)
    X_test = cv.transform(reviews_test_clean)

    target = [0 if i < 12500 else 1 for i in range(25000)]

    X_train, X_val, y_train, y_val = train_test_split(
        X, target, train_size=0.75
    )
    c_values = [0.05, 0.25, 0.5, 1]
    for c in c_values:
        final_model = LogisticRegression(C=c)
        final_model.fit(X, target)
        print("Logistic Regression with c = " + str(c) +
              " accuracy is %s"
              % accuracy_score(target, final_model.predict(X_test)))

    # Prepare for the RNN models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(torch.cuda.is_available())
    data = pd.read_csv('./data/allData.csv', lineterminator='\n', encoding='utf-8')
    data.head()
    lang = Lang()

    for sentence_data in data["review"].values.tolist():
        sentence_data = normalizeString(sentence_data)
        lang.addSentence(sentence_data)

    data_count = np.array(list(lang.word2count.values()))
    np.median(data_count), np.mean(data_count), np.max(data_count)

    less_count = 0
    total_count = 0

    for _, count in lang.word2count.items():
        if count < 50:
            less_count = less_count + count
        total_count = total_count + count

    lang_process = Lang()

    for word, count in lang.word2count.items():
        if count >= 50:
            lang_process.word2index[word] = lang_process.n_words
            lang_process.word2count[word] = count
            lang_process.index2word[lang_process.n_words] = word
            lang_process.n_words = lang_process.n_words + 1

    input_tensor = [[convertWord2index(s) for s in normalizeString(es).split(' ')] for es in
                    data["review"].values.tolist()]

    sentence_length = [len(t) for t in input_tensor]

    input_tensor = [pad_sequences(x, 1000) for x in input_tensor]

    index2emotion = {0: 'Negative\r', 1: 'Positive\r'}
    emotion2index = {'Negative\r': 0, 'Positive\r': 1}
    target_tensor = [emotion2index.get(s) for s in data['label\r'].values.tolist()]

    END = int(len(input_tensor) * 0.5)
    input_tensor_train = torch.from_numpy(np.array(input_tensor[:END]))
    target_tensor_train = torch.from_numpy(np.array(target_tensor[:END])).long()
    input_tensor_test = torch.from_numpy(np.array(input_tensor[END:]))
    target_tensor_test = torch.from_numpy(np.array(target_tensor[END:])).long()

    train_dataset = Data.TensorDataset(input_tensor_train, target_tensor_train)  # 训练样本
    test_dataset = Data.TensorDataset(input_tensor_test, target_tensor_test)  # 测试样本

    MINIBATCH_SIZE = 64

    train_loader = Data.DataLoader(
        dataset=train_dataset,
        batch_size=MINIBATCH_SIZE,
        shuffle=True,
        num_workers=2  # set multi-work num read data
    )

    test_loader = Data.DataLoader(
        dataset=test_dataset,
        batch_size=MINIBATCH_SIZE,
        shuffle=True,
        num_workers=2  # set multi-work num read data
    )

    # Simple RNN
    print("====================Simple RNN====================")
    vocab_inp_size = len(lang_process.word2index) + 2
    embedding_dim = 256
    hidden_units = 512
    target_size = 2  # 2  emotion

    vocab_inp_size = len(lang_process.word2index) + 2
    embedding_dim = 256
    hidden_units = 512
    target_size = 2
    num_layers = [1]

    for layers in num_layers:
        modelRNN = EmotionRNN(vocab_inp_size, embedding_dim, hidden_units, MINIBATCH_SIZE, target_size).to(device)
        models = {}

        models['RNN'] = modelRNN
        for key, model in models.items():
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.SGD(model.parameters(), lr=0.001)

            num_epochs = 30
            for epoch in range(num_epochs):
                model.train()
                start = time.time()

                train_total_loss = 0
                train_total_accuracy = 0

                ### Training
                for batch, (inp, targ) in enumerate(train_loader):
                    predictions = model(inp.to(device))
                    loss = criterion(predictions, targ.to(device))
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    batch_loss = (loss / int(targ.size(0)))
                    batch_accuracy = accuracy(predictions, targ.to(device))

                    train_total_loss = train_total_loss + batch_loss
                    train_total_accuracy = train_total_accuracy + batch_accuracy

                    if batch % 100 == 0:
                        record_train_accuracy = train_total_accuracy.cpu().detach().numpy() / (batch + 1)
                        print('Epoch {} Batch {} Accuracy {:.4f}. Loss {:.4f}'.format(epoch + 1,
                                                                                      batch,
                                                                                      train_total_accuracy.cpu().detach().numpy() / (
                                                                                                  batch + 1),
                                                                                      train_total_loss.cpu().detach().numpy() / (
                                                                                                  batch + 1)))

                print('------------')
                model.eval()
                test_total_accuracy = 0
                for batch, (input_data, target_data) in enumerate(test_loader):
                    predictions = model(input_data.to(device))
                    batch_accuracy = accuracy(predictions, target_data.to(device))
                    test_total_accuracy = test_total_accuracy + batch_accuracy
                print('Test : Model {}, Epoch {} Accuracy {:.4f}'.format(key, epoch + 1,
                                                                         test_total_accuracy.cpu().detach().numpy() / (
                                                                                     batch + 1)))
                record_test_accuracy = test_total_accuracy.cpu().detach().numpy() / (batch + 1)
                end = time.time()
                epoch_mins, epoch_secs = epoch_time(start, end)
                print('Epoch time : {}mins {}secs'.format(epoch_mins, epoch_secs))
                print('============')

    # LSTM
    print("====================LSTM====================")
    vocab_inp_size = len(lang_process.word2index) + 2
    embedding_dim = 256
    hidden_units = 512
    target_size = 2  # 2  emotion
    num_layers = [2]

    vocab_inp_size = len(lang_process.word2index) + 2
    embedding_dim = 256
    hidden_units = 512
    target_size = 2
    num_layers = [2]

    for layers in num_layers:
        # Test model
        modelLSTM = EmotionLSTM(vocab_inp_size, embedding_dim, hidden_units, MINIBATCH_SIZE, target_size, layers).to(
            device)
        models = {}

        models['LSTM'] = modelLSTM
        # models['GRU'] = modelGRU
        for key, model in models.items():
            # Define the loss function and optimizer
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)

            # Start training
            num_epochs = 30
            for epoch in range(num_epochs):
                model.train()
                start = time.time()

                train_total_loss = 0  # Record the average loss in an entire epoch
                train_total_accuracy = 0  # Record the average accuracy in an entire epoch

                ### Training
                for batch, (inp, targ) in enumerate(train_loader):
                    predictions = model(inp.to(device))
                    # Error in calculation
                    loss = criterion(predictions, targ.to(device))
                    # Reverse propagation to modify the weight
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    # Record Loss decreases and accuracy improves
                    batch_loss = (loss / int(targ.size(0)))  # Record a bacth loss
                    batch_accuracy = accuracy(predictions, targ.to(device))

                    train_total_loss = train_total_loss + batch_loss
                    train_total_accuracy = train_total_accuracy + batch_accuracy

                    if batch % 100 == 0:
                        record_train_accuracy = train_total_accuracy.cpu().detach().numpy() / (batch + 1)
                        print('Layer {} Epoch {} Batch {} Accuracy {:.4f}. Loss {:.4f}'.format(layers, epoch + 1,
                                                                                               batch,
                                                                                               train_total_accuracy.cpu().detach().numpy() / (
                                                                                                           batch + 1),
                                                                                               train_total_loss.cpu().detach().numpy() / (
                                                                                                           batch + 1)))
                # Each epoch is used to calculate the accuracy of the test
                print('------------')
                model.eval()
                test_total_accuracy = 0
                for batch, (input_data, target_data) in enumerate(test_loader):
                    predictions1 = model(input_data.to(device))
                    batch_accuracy1 = accuracy(predictions1, target_data.to(device))
                    test_total_accuracy = test_total_accuracy + batch_accuracy1
                print('Test : Lay {}, Model {}, Epoch {} Accuracy {:.4f}'.format(layers, key, epoch + 1,
                                                                                 test_total_accuracy.cpu().detach().numpy() / (
                                                                                             batch + 1)))
                record_test_accuracy = test_total_accuracy.cpu().detach().numpy() / (batch + 1)
                end = time.time()
                epoch_mins, epoch_secs = epoch_time(start, end)
                print('Epoch time : {}mins {}secs'.format(epoch_mins, epoch_secs))
                print('============')

    # GRU
    print("====================GRU====================")
    vocab_inp_size = len(lang_process.word2index) + 2
    embedding_dim = 256
    hidden_units = 512
    target_size = 2  # 2  emotion
    num_layers = [2]

    vocab_inp_size = len(lang_process.word2index) + 2
    embedding_dim = 256
    hidden_units = 512
    target_size = 2
    num_layers = [2]

    for layers in num_layers:
        modelGRU = EmotionGRU(vocab_inp_size, embedding_dim, hidden_units, MINIBATCH_SIZE, target_size, layers).to(
            device)
        models = {}

        models['GRU'] = modelGRU
        for key, model in models.items():
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)

            num_epochs = 30
            for epoch in range(num_epochs):
                model.train()
                start = time.time()

                train_total_loss = 0
                train_total_accuracy = 0

                ### Training
                for batch, (inp, targ) in enumerate(train_loader):
                    predictions = model(inp.to(device))
                    loss = criterion(predictions, targ.to(device))
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    batch_loss = (loss / int(targ.size(0)))
                    batch_accuracy = accuracy(predictions, targ.to(device))

                    train_total_loss = train_total_loss + batch_loss
                    train_total_accuracy = train_total_accuracy + batch_accuracy

                    if batch % 100 == 0:
                        record_train_accuracy = train_total_accuracy.cpu().detach().numpy() / (batch + 1)
                        print('Epoch {} Batch {} Accuracy {:.4f}. Loss {:.4f}'.format(epoch + 1,
                                                                                      batch,
                                                                                      train_total_accuracy.cpu().detach().numpy() / (
                                                                                                  batch + 1),
                                                                                      train_total_loss.cpu().detach().numpy() / (
                                                                                                  batch + 1)))

                print('------------')
                model.eval()
                test_total_accuracy = 0
                for batch, (input_data, target_data) in enumerate(test_loader):
                    predictions = model(input_data.to(device))
                    batch_accuracy = accuracy(predictions, target_data.to(device))
                    test_total_accuracy = test_total_accuracy + batch_accuracy
                print('Test : Lay {}, Model {}, Epoch {} Accuracy {:.4f}'.format(layers, key, epoch + 1,
                                                                                 test_total_accuracy.cpu().detach().numpy() / (
                                                                                             batch + 1)))
                record_test_accuracy = test_total_accuracy.cpu().detach().numpy() / (batch + 1)
                end = time.time()
                epoch_mins, epoch_secs = epoch_time(start, end)
                print('Epoch time : {}mins {}secs'.format(epoch_mins, epoch_secs))
                print('============')

    # Bi-LSTM
    print("====================Bi-LSTM====================")
    # Model hyperparameter
    vocab_inp_size = len(lang_process.word2index) + 2
    embedding_dim = 256
    hidden_units = 512
    target_size = 2
    num_layers = [2]

    for layers in num_layers:
        # Test model
        modelLSTM = Bi_EmotionLSTM(vocab_inp_size, embedding_dim, hidden_units, MINIBATCH_SIZE, target_size, layers).to(
            device)
        models = {}

        models['LSTM'] = modelLSTM
        # models['GRU'] = modelGRU
        for key, model in models.items():
            # Define the loss function and optimizer
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)

            # Start training
            num_epochs = 50
            for epoch in range(num_epochs):
                model.train()
                start = time.time()

                train_total_loss = 0  # Record the average loss in an entire epoch
                train_total_accuracy = 0  # Record the average accuracy in an entire epoch

                ### Training
                for batch, (inp, targ) in enumerate(train_loader):
                    predictions = model(inp.to(device))
                    # Error in calculation
                    loss = criterion(predictions, targ.to(device))
                    # Reverse propagation to modify the weight
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    # Record Loss decreases and accuracy improves
                    batch_loss = (loss / int(targ.size(0)))  # Record a bacth loss
                    batch_accuracy = accuracy(predictions, targ.to(device))

                    train_total_loss = train_total_loss + batch_loss
                    train_total_accuracy = train_total_accuracy + batch_accuracy

                    if batch % 100 == 0:
                        record_train_accuracy = train_total_accuracy.cpu().detach().numpy() / (batch + 1)
                        print('Layer {} Epoch {} Batch {} Accuracy {:.4f}. Loss {:.4f}'.format(layers, epoch + 1,
                                                                                               batch,
                                                                                               train_total_accuracy.cpu().detach().numpy() / (
                                                                                                           batch + 1),
                                                                                               train_total_loss.cpu().detach().numpy() / (
                                                                                                           batch + 1)))
                # Each epoch is used to calculate the accuracy of the test
                print('------------')
                model.eval()
                test_total_accuracy = 0
                for batch, (input_data, target_data) in enumerate(test_loader):
                    predictions1 = model(input_data.to(device))
                    batch_accuracy1 = accuracy(predictions1, target_data.to(device))
                    test_total_accuracy = test_total_accuracy + batch_accuracy1
                print('Test : Lay {}, Model {}, Epoch {} Accuracy {:.4f}'.format(layers, key, epoch + 1,
                                                                                 test_total_accuracy.cpu().detach().numpy() / (
                                                                                             batch + 1)))
                record_test_accuracy = test_total_accuracy.cpu().detach().numpy() / (batch + 1)
                end = time.time()
                epoch_mins, epoch_secs = epoch_time(start, end)
                print('Epoch time : {}mins {}secs'.format(epoch_mins, epoch_secs))
                print('============')

    # Bi-GRU
    print("====================Bi-GRU====================")
    vocab_inp_size = len(lang_process.word2index) + 2
    embedding_dim = 256
    hidden_units = 512
    target_size = 2
    num_layers = [2]

    for layers in num_layers:
        modelGRU = Bi_EmotionGRU(vocab_inp_size, embedding_dim, hidden_units, MINIBATCH_SIZE, target_size, layers).to(
            device)
        models = {}

        models['GRU'] = modelGRU
        for key, model in models.items():
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)

            num_epochs = 30
            for epoch in range(num_epochs):
                model.train()
                start = time.time()

                train_total_loss = 0
                train_total_accuracy = 0

                for batch, (inp, targ) in enumerate(train_loader):
                    predictions = model(inp.to(device))
                    loss = criterion(predictions, targ.to(device))
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    batch_loss = (loss / int(targ.size(0)))
                    batch_accuracy = accuracy(predictions, targ.to(device))

                    train_total_loss = train_total_loss + batch_loss
                    train_total_accuracy = train_total_accuracy + batch_accuracy

                    if batch % 100 == 0:
                        record_train_accuracy = train_total_accuracy.cpu().detach().numpy() / (batch + 1)
                        print('Epoch {} Batch {} Accuracy {:.4f}. Loss {:.4f}'.format(epoch + 1,
                                                                                      batch,
                                                                                      train_total_accuracy.cpu().detach().numpy() / (
                                                                                                  batch + 1),
                                                                                      train_total_loss.cpu().detach().numpy() / (
                                                                                                  batch + 1)))

                print('------------')
                model.eval()
                test_total_accuracy = 0
                for batch, (input_data, target_data) in enumerate(test_loader):
                    predictions = model(input_data.to(device))
                    batch_accuracy = accuracy(predictions, target_data.to(device))
                    test_total_accuracy = test_total_accuracy + batch_accuracy
                print('Test : Lay {}, Model {}, Epoch {} Accuracy {:.4f}'.format(layers, key, epoch + 1,
                                                                                 test_total_accuracy.cpu().detach().numpy() / (
                                                                                             batch + 1)))
                record_test_accuracy = test_total_accuracy.cpu().detach().numpy() / (batch + 1)
                end = time.time()
                epoch_mins, epoch_secs = epoch_time(start, end)
                print('Epoch time : {}mins {}secs'.format(epoch_mins, epoch_secs))
                print('============')

    print("Program End!")