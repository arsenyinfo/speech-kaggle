from keras.models import Model
from keras.layers import Dense, Bidirectional, Input, BatchNormalization, Conv2D, MaxPooling2D, Dropout, LSTM
from keras.layers import GlobalAveragePooling1D, GlobalAveragePooling2D, TimeDistributed, Concatenate, Lambda
from keras import backend as K


def lstm_block(x):
    x = BatchNormalization(axis=-1, momentum=0.99, epsilon=1e-3, center=True, scale=True)(x)
    x = Bidirectional(LSTM(128, return_sequences=True, activation='tanh', recurrent_dropout=.1),
                      merge_mode='concat')(x)
    x = Bidirectional(LSTM(64, return_sequences=True, activation='tanh', recurrent_dropout=.1),
                      merge_mode='concat')(x)
    x = TimeDistributed(Dense(64, activation='relu'))(x)
    return BatchNormalization()(GlobalAveragePooling1D()(x))


def conv_block(x):
    x = Lambda(lambda x: K.expand_dims(x, axis=-1))(x)
    x = Conv2D(32, kernel_size=3, activation='relu')(x)
    x = Conv2D(32, kernel_size=3, activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(rate=0.5)(x)
    x = Conv2D(64, kernel_size=3, activation='relu', padding='same')(x)
    x = Conv2D(64, kernel_size=3, activation='relu')(x)
    x = Dropout(rate=0.5)(x)
    return BatchNormalization()(GlobalAveragePooling2D()(x))


def naivenet(shape1, shape2, output_dim):
    inp1 = Input(name='the_input', shape=shape1)
    x1 = lstm_block(inp1)

    inp2 = Input(name='the_input_2', shape=shape2)
    x2 = lstm_block(inp2)

    x3 = conv_block(inp1)
    x4 = conv_block(inp2)

    x = Concatenate()([x1, x2, x3, x4])
    x = Dropout(.5)(x)

    out = Dense(output_dim, activation='softmax')(x)
    model = Model(inputs=[inp1, inp2], outputs=out)

    return model
