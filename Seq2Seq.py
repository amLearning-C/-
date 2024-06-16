import jieba
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataPrepare import *
import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow import keras
import matplotlib.pyplot as plt
import os

callbacks_list = [
    keras.callbacks.ModelCheckpoint(filepath='text_gen.keras', monitor='loss', save_best_only=True),
    keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=1),
    keras.callbacks.EarlyStopping(monitor='loss', patience=3),
]

class SeqToSeq(nn.Module):
    def __init__(self, len_token, embedding_size):
        super(SeqToSeq, self).__init__()
        self.encode = nn.Embedding(len_token, embedding_size)
        self.lstm = nn.LSTM(embedding_size, embedding_size, 2, batch_first=True)
        self.decode = nn.Sequential(
            nn.Linear(embedding_size, len_token),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        print(x.shape)
        em = self.encode(x).unsqueeze(dim=1)
        print(em.shape)
        mid, _ = self.lstm(em)
        print(mid[:,0,:].shape)
        res = self.decode(mid[:, 0, :])
        print(res.shape)
        return res

def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def train(x, y, tokens, tokens_indices, epochs=5, file_name="model"):
    x = np.asarray(x)
    y = np.asarray(y)
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.shuffle(buffer_size=4096)
    dataset = dataset.batch(64)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    model = models.Sequential([
        layers.Embedding(len(tokens), 256),
        layers.LSTM(256),
        layers.Dense(len(tokens), activation='softmax')
    ])

    optimizer = optimizers.RMSprop(learning_rate=0.1)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)

    history = model.fit(dataset, epochs=epochs, callbacks=callbacks_list)

    # 保存模型
    model_file_name = f"{file_name}_{epochs}_epochs.keras"
    model.save(model_file_name)
    print(f"Model saved as {model_file_name}")

    return model, history

def generate_text(model, tokens, tokens_indices, initial_text, length=100, temperatures=[0.5, 1.0, 1.2]):
    text = initial_text
    print(text, end='')
    for temperature in temperatures:
        text_cut = list(jieba.cut(text))[:60]
        print('\n temperature: ', temperature)
        print(''.join(text_cut), end='')
        for i in range(length):
            sampled = np.zeros((1, 60))
            for idx, token in enumerate(text_cut):
                if token in tokens_indices:
                    sampled[0, idx] = tokens_indices[token]
            preds = model.predict(sampled, verbose=0)[0]
            next_index = sample(preds, temperature)
            next_token = tokens[next_index]
            print(next_token, end='')

            text_cut = text_cut[1: 60] + [next_token]

def plot_loss(history):
    plt.plot(history.history['loss'], label='Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    file = DATA_PATH + '越女剑.txt'
    d = get_single_corpus(file)
    _x, _y, _tokens, _tokens_indices = get_dataset(d)
    model, history = train(_x, _y, _tokens, _tokens_indices, epochs=40, file_name=os.path.splitext(os.path.basename(file))[0])
    plot_loss(history)
    initial_text = '第五日上，文种来到范府拜访，见范府掾吏面有忧色，问道：“范大夫多日不见，大王颇为挂念，命我前来探望，莫非范大夫身子不适么？”那掾吏道：“回禀文大夫：范大夫身子并无不适，只是……只是……”文种道：“只是怎样？”'
    generate_text(model, _tokens, _tokens_indices, initial_text)
