import os
import argparse
import joblib
import numpy as np
import math
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger, LearningRateScheduler, TerminateOnNaN, LambdaCallback, Callback
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, Activation
from tensorflow.keras.layers import Input, MaxPooling2D, Dropout, Flatten
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Layer

weight_decay = 1e-4

PARAMS = {
    "arcface": [1., 0.5, 0., 0.],
    "cosface": [1., 0., .35, 0.],
    "cosface_extend": [1., 0., .35, None],
    "sphereface": [1.35, 0., 0., 0.],
    "sphereface_extend": [1.35, 0., 0., None],
    "normface": [1., 0., 0., 0.]
    }

class CosineAnnealingScheduler(Callback):
    """Cosine annealing scheduler.
    """

    def __init__(self, T_max, eta_max, eta_min=0, verbose=0):
        super(CosineAnnealingScheduler, self).__init__()
        self.T_max = T_max
        self.eta_max = eta_max
        self.eta_min = eta_min
        self.verbose = verbose

    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, 'lr'):
            raise ValueError('Optimizer must have a "lr" attribute.')
        lr = self.eta_min + (self.eta_max - self.eta_min) * (1 + math.cos(math.pi * epoch / self.T_max)) / 2
        K.set_value(self.model.optimizer.lr, lr)
        if self.verbose > 0:
            print('\nEpoch %05d: CosineAnnealingScheduler setting learning '
                  'rate to %s.' % (epoch + 1, lr))

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['lr'] = K.get_value(self.model.optimizer.lr)

class CosSerisLayer(Layer):
    '''Input: Feature Scale s, Margin Parameter m in Eq. 3, Class Number n, Ground-Truth ID gt.
        1. x = mx.symbol.L2Normalization (x, mode = ’instance’)
        2. W = mx.symbol.L2Normalization (W, mode = ’instance’)
        3. fc7 = mx.sym.FullyConnected (data = x, weight = W, no bias = True, num hidden = n)
        4. original target logit = mx.sym.pick (fc7, gt, axis = 1)
        5. theta = mx.sym.arccos (original target logit)
        6. marginal target logit = mx.sym.cos (theta + m)
        7. one hot = mx.sym.one hot (gt, depth = n, on value = 1.0, off value = 0.0)
        8. fc7 = fc7 + mx.sym.broadcast mul (one hot, mx.sym.expand dims (marginal target logit - original target logit, 1))
        9. fc7 = fc7 * s#

       cos(m1 * x + m2) + m3
    '''

    def __init__(self, output_dim, m1, m2, m3, m4=None, scale=30., regularizer=None, **kwargs):
        super(CosSerisLayer, self).__init__(**kwargs)
        self.output_dim = output_dim
        self.m1, self.m2, self.m3 = m1, m2, m3
        self.m4 = m2 / 4 if m4 is None else m4
        self.scale = scale
        self.regularizer = regularizers.get(regularizer)


    def build(self, input_shape):
        print("input_shape", input_shape)
        super(CosSerisLayer, self).build(input_shape)
        self.input_dim = input_shape[0][-1]
        self.w = self.add_weight(
                name="W",
                shape=(self.input_dim, self.output_dim),
                initializer='glorot_uniform',
                trainable=True,
                regularizer=self.regularizer
            )

    def call(self, inputs):
        x, onehot = inputs
        norm_x = tf.nn.l2_normalize(x, axis=-1)
        print(onehot.shape)

        #onehot = tf.squeeze(tf.one_hot(tf.cast(labels, tf.int32), self.output_dim), axis=[1])
        W = tf.nn.l2_normalize(self.w, axis=0)
        fc = norm_x @ W
        if self.m1 != 1. or self.m2 or self.m3 or self.m4:
            theta = tf.acos(K.clip(fc, -1 + K.epsilon(), 1 - K.epsilon()))
            fc = tf.where(onehot > 0, tf.cos(K.clip(self.m1 * theta + self.m2, 0., np.pi)), tf.cos(K.clip(theta - self.m4, 0., np.pi)))

        return tf.nn.softmax(self.scale * fc)

def vgg_block(x, filters, layers):
    for _ in range(layers):
        x = Conv2D(filters, (3, 3), padding='same', kernel_initializer='he_normal',
                    kernel_regularizer=regularizers.l2(weight_decay))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
    return x

def get_transform_layer(args):
    arch = args.arch
    clas_nums, scale = 10, 30
    if arch in PARAMS:
        m1, m2, m3, m4 = PARAMS.get(arch)
    else:
        raise Exception("fata arch: %s"%args.arch)
    return CosSerisLayer(10, m1, m2, m3, m4, scale, regularizer=regularizers.l2(weight_decay))

def vgg8(args):
    input = Input(shape=(28, 28, 1))
    y = Input(shape=(10,))

    x = vgg_block(input, 16, 2)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = vgg_block(x, 32, 2)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = vgg_block(x, 64, 2)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Flatten()(x)
    x = Dense(args.num_features, kernel_initializer='he_normal',
                kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = BatchNormalization()(x)

    output = get_transform_layer(args)([x, y])
    return Model(inputs=[input, y], outputs=[output])

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=None,
                        help='model name: (default: arch+timestamp)')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='arcface',
                        choices=PARAMS.keys())
    parser.add_argument('--num-features', default=3, type=int,
                        help='dimention of embedded features')
    parser.add_argument('--scheduler', default='CosineAnnealing',
                        choices=['CosineAnnealing', 'None'],
                        help='scheduler: ' +
                            ' | '.join(['CosineAnnealing', 'None']) +
                            ' (default: CosineAnnealing)')
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch-size', default=128, type=int,
                        metavar='N', help='mini-batch size (default: 128)')
    parser.add_argument('--optimizer', default='SGD',
                        choices=['Adam', 'SGD'],
                        help='loss: ' +
                            ' | '.join(['Adam', 'SGD']) +
                            ' (default: Adam)')
    parser.add_argument('--lr', '--learning-rate', default=1e-1, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--min-lr', default=1e-3, type=float,
                        help='minimum learning rate')
    parser.add_argument('--momentum', default=0.5, type=float)
    parser.add_argument('--draw', action='store_true')
    parser.add_argument('--pretrained', action='store_true')

    args = parser.parse_args()

    return args

def main():
    args = parse_args()

    # add model name to args
    args.name = 'mnist_%s_%dd' %(args.arch, args.num_features)

    os.makedirs('models/%s' %args.name, exist_ok=True)

    print('Config -----')
    for arg in vars(args):
        print('%s: %s' %(arg, getattr(args, arg)))
    print('------------')

    joblib.dump(args, 'models/%s/args.pkl' %args.name)

    with open('models/%s/args.txt' %args.name, 'w+') as f:
        for arg in vars(args):
            print('%s: %s' %(arg, getattr(args, arg)), file=f)

    (X, y), (X_test, y_test) = mnist.load_data()
    y_test_org = y_test

    X = X[:, :, :, np.newaxis].astype('float32') / 255
    X_test = X_test[:, :, :, np.newaxis].astype('float32') / 255

    y = keras.utils.to_categorical(y, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    if args.optimizer == 'SGD':
        optimizer = SGD(lr=args.lr, momentum=args.momentum)
    elif args.optimizer == 'Adam':
        optimizer = Adam(lr=args.lr)

    model = vgg8(args)
    model.compile(loss='categorical_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])
    model.summary()

    callbacks = [
        ModelCheckpoint(os.path.join('models', args.name, 'model.hdf5'),
            verbose=1, save_best_only=True),
        CSVLogger(os.path.join('models', args.name, 'log.csv')),
        TerminateOnNaN()]

    if args.scheduler == 'CosineAnnealing':
        callbacks.append(CosineAnnealingScheduler(T_max=args.epochs, eta_max=args.lr, eta_min=args.min_lr, verbose=1))

    if args.arch in PARAMS and not args.pretrained:
        model.fit([X, y], y, validation_data=([X_test, y_test], y_test),
            batch_size=args.batch_size,
            epochs=args.epochs,
            callbacks=callbacks,
            verbose=1)
    if args.draw:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        model.load_weights(os.path.join('models/%s/model.hdf5' %args.name))
        print(model.layers[-3:])
        model = Model(model.inputs[0], model.layers[-3].output)
        embedding = model.predict(X_test, verbose=1)
        embedding /= np.linalg.norm(embedding, axis=1, keepdims=True)
        print(embedding.shape)

        fig = plt.figure()
        axes = Axes3D(fig)
        for c in range(len(np.unique(y_test_org))):
            axes.plot(embedding[y_test_org==c, 0], embedding[y_test_org==c, 1], embedding[y_test_org==c, 2], '.', alpha=0.1)
        plt.title('%s'%args.arch)
        plt.savefig('%s.png'%args.arch)

if __name__ == "__main__":
   main()
