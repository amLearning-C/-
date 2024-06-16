import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers
from dataPrepare import *  # 包含了数据准备的相关函数
import matplotlib.pyplot as plt

callbacks_list = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath='text_gen.keras',
        monitor='loss',
        save_best_only=True,
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='loss',
        factor=0.5,
        patience=1,
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor='loss',
        patience=3,
    ),
    tf.keras.callbacks.LambdaCallback(
        on_epoch_end=lambda epoch, logs: print(f'Epoch {epoch + 1}, Loss: {logs["loss"]}')
    ),
]

def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

class TransformerEncoder(layers.Layer):
    def __init__(self, num_heads, key_dim, ff_dim, rate=0.1):
        super(TransformerEncoder, self).__init__()
        self.mha = layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)
        self.ffn = keras.Sequential([
            layers.Dense(ff_dim, activation='relu'),
            layers.Dense(key_dim)
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, x, training):
        attn_output = self.mha(x, x)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

def build_transformer_model(vocab_size, d_model, num_heads, ff_dim, maxlen):
    inputs = layers.Input(shape=(maxlen,))
    embedding_layer = layers.Embedding(input_dim=vocab_size, output_dim=d_model)
    x = embedding_layer(inputs)

    transformer_block = TransformerEncoder(num_heads=num_heads, key_dim=d_model, ff_dim=ff_dim)
    x = transformer_block(x, training=True)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.1)(x)
    outputs = layers.Dense(vocab_size, activation="softmax")(x)

    return models.Model(inputs=inputs, outputs=outputs)

def train(x, y, tokens, tokens_indices, text, epochs=10):
    x = np.asarray(x)
    y = np.asarray(y)
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.shuffle(buffer_size=4096)
    dataset = dataset.batch(128)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    maxlen = x.shape[1]
    model = build_transformer_model(vocab_size=len(tokens), d_model=256, num_heads=8, ff_dim=512, maxlen=maxlen)

    optimizer = optimizers.Adam(learning_rate=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)

    history = model.fit(dataset, epochs=epochs, callbacks=callbacks_list)

    # 训练完成后生成文本
    generate_text_after_training(model, tokens_indices, tokens, text)

    # 绘制损失曲线
    plot_loss(history)

def generate_text_after_training(model, tokens_indices, tokens, text):
    for temperature in [0.2, 0.5, 1.0, 1.2]:
        text_cut = list(jieba.cut(text))[:60]
        print('\n temperature: ', temperature)
        print(''.join(text_cut), end='')
        for i in range(100):
            sampled = np.zeros((1, 60))
            for idx, token in enumerate(text_cut):
                if token in tokens_indices:
                    sampled[0, idx] = tokens_indices[token]
            preds = model.predict(sampled, verbose=0)[0]
            next_index = sample(preds, temperature=temperature)
            next_token = tokens[next_index]
            print(next_token, end='')

            text_cut = text_cut[1: 60] + [next_token]

def plot_loss(history):
    plt.plot(history.history['loss'], label='Train Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    file = DATA_PATH + '越女剑.txt'  # 修改为实际文件路径
    d = get_single_corpus(file)
    _x, _y, _tokens, _tokens_indices = get_dataset(d)
    text = '第五日上，文种来到范府拜访，见范府掾吏面有忧色，问道：“范大夫多日不见，大王颇为挂念，命我前来探望，莫非范大夫身子不适么？”那掾吏道：“回禀文大夫：范大夫身子并无不适，只是……只是……”文种道：“只是怎样？”'
    train(_x, _y, _tokens, _tokens_indices,epochs=80,text=text)
