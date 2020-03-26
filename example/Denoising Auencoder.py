### Denoising Autoencoder



### 합수 및 모듈
import tensorflow as tf

import os, sys
import numpy as np
import tensorflow as tf


# 일관된 출력을 위해 유사난수 초기화
def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)


import matplotlib
import matplotlib.pyplot as plt
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12


### 한글출력
# matplotlib.rc('font', family='AppleGothic')  # MacOS
matplotlib.rc('font', family='Malgun Gothic')  # Windows
# matplotlib.rc('font', family='NanumBarunGothic') # Linux
plt.rcParams['axes.unicode_minus'] = False


## 그래프 함수
def plot_image(image, shape=[28, 28]):
    plt.imshow(image.reshape(shape), cmap="Greys", interpolation="nearest")
    plt.axis("off")


def plot_multiple_images(images, n_rows, n_cols, pad=2):
    images = images - images.min()  # 최소값을 0으로 만들어 패딩이 하얗게 보이도록 합니다.
    w, h = images.shape[1:]
    image = np.zeros(((w + pad) * n_rows + pad, (h + pad) * n_cols + pad))
    for y in range(n_rows):
        for x in range(n_cols):
            image[(y * (h + pad) + pad):(y * (h + pad) + pad + h), (x * (w + pad) + pad):(x * (w + pad) + pad + w)] = \
            images[y * n_cols + x]
    plt.imshow(image, cmap="Greys", interpolation="nearest")
    plt.axis("off")


### 3.1 텐서플로로 stacked 오토인코더 구현
# Mnist Data Load
(train_x, train_y), (test_x, test_y) = tf.keras.datasets.mnist.load_data()
train_x = train_x.astype(np.float32).reshape(-1, 28*28) / 255.0
test_x = test_x.astype(np.float32).reshape(-1, 28*28) / 255.0
train_y = train_y.astype(np.int32)
test_y = test_y.astype(np.int32)
valid_x, train_x = train_x[:5000], train_x[5000:]
valid_y, train_y = train_y[:5000], train_y[5000:]

# Mini-batch
def shuffle_batch(features, labels, batch_size):
    rnd_idx = np.random.permutation(len(features))
    n_batches = len(features) // batch_size
    for batch_idx in np.array_split(rnd_idx, n_batches):
        batch_x, batch_y = features[batch_idx], labels[batch_idx]
        yield batch_x, batch_y



def show_reconstructed_digits(X, outputs, model_path=None, n_test_digits=2):
    with tf.Session() as sess:
        if model_path:
            saver.restore(sess, model_path)
        outputs_val = outputs.eval(feed_dict={inputs: test_x[:n_test_digits]})

    fig = plt.figure(figsize=(10, 4))
    for digit_index in range(n_test_digits):
        plt.subplot(n_test_digits, 2, digit_index * 2 + 1)
        plot_image(test_x[digit_index])
        plt.subplot(n_test_digits, 2, digit_index * 2 + 2)
        plot_image(outputs_val[digit_index])


########################################################################################################################
reset_graph()

################
# layer params #
################
noise_level = 1.0
n_inputs = 28 * 28
n_hidden1 = 300
n_hidden2 = 150  # coding units
n_hidden3 = n_hidden1
n_outputs = n_inputs

################
# train params #
################
learning_rate = 0.01
n_epochs = 10
batch_size = 150

# denoising autoencoder
inputs = tf.placeholder(tf.float32, shape=[None, n_inputs])
# add gaussian noise
inputs_noisy = inputs + noise_level * tf.random_normal(tf.shape(inputs))

hidden1 = tf.layers.dense(inputs_noisy, n_hidden1, activation=tf.nn.relu, name='hidden1')
hidden2 = tf.layers.dense(hidden1, n_hidden2, activation=tf.nn.relu, name='hidden2')
hidden3 = tf.layers.dense(hidden2, n_hidden3, activation=tf.nn.relu, name='hidden3')
outputs = tf.layers.dense(hidden3, n_outputs, name='outputs')

# loss
reconstruction_loss = tf.losses.mean_squared_error(labels=inputs, predictions=outputs)
# optimizer
train_op = tf.train.AdamOptimizer(learning_rate).minimize(reconstruction_loss)

# saver
saver = tf.train.Saver()

# Train
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    n_batches = len(train_x) // batch_size
    for epoch in range(n_epochs):
        for iteration in range(n_batches):
            print("\r{}%".format(100 * iteration // n_batches), end="")
            sys.stdout.flush()
            batch_x, batch_y = next(shuffle_batch(train_x, train_y, batch_size))
            sess.run(train_op, feed_dict={inputs: batch_x})
        loss_train = reconstruction_loss.eval(feed_dict={inputs: batch_x})
        print('\repoch : {}, Train MSE : {:.5f}'.format(epoch, loss_train))
    saver.save(sess, './model/my_model_stacked_denoising_gaussian.ckpt')



show_reconstructed_digits(inputs, outputs, './model/my_model_stacked_denoising_gaussian.ckpt')


########################################################################################################################
########################################################################################################################
reset_graph()

################
# layer params #
################
noise_level = 1.0
n_inputs = 28 * 28
n_hidden1 = 300
n_hidden2 = 150  # coding units
n_hidden3 = n_hidden1
n_outputs = n_inputs

################
# train params #
################
dropout_rate = 0.3
learning_rate = 0.01
n_epochs = 10
batch_size = 150

training = tf.placeholder_with_default(False, shape=(), name='training')

# denoising autoencoder
inputs = tf.placeholder(tf.float32, shape=[None, n_inputs])
# add dropout
inputs_drop = tf.layers.dropout(inputs, dropout_rate, training=training)

hidden1 = tf.layers.dense(inputs_drop, n_hidden1, activation=tf.nn.relu, name='hidden1')
hidden2 = tf.layers.dense(hidden1, n_hidden2, activation=tf.nn.relu, name='hidden2')
hidden3 = tf.layers.dense(hidden2, n_hidden3, activation=tf.nn.relu, name='hidden3')
outputs = tf.layers.dense(hidden3, n_outputs, name='outputs')

# loss
reconstruction_loss = tf.losses.mean_squared_error(labels=inputs, predictions=outputs)
# optimizer
train_op = tf.train.AdamOptimizer(learning_rate).minimize(reconstruction_loss)

# saver
saver = tf.train.Saver()

# Train
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    n_batches = len(train_x) // batch_size
    for epoch in range(n_epochs):
        for iteration in range(n_batches):
            print("\r{}%".format(100 * iteration // n_batches), end="")
            sys.stdout.flush()
            batch_x, batch_y = next(shuffle_batch(train_x, train_y, batch_size))
            sess.run(train_op, feed_dict={inputs: batch_x})
        loss_train = reconstruction_loss.eval(feed_dict={inputs: batch_x})
        print('\repoch : {}, Train MSE : {:.5f}'.format(epoch, loss_train))
    saver.save(sess, './model/my_model_stacked_denoising_dropout.ckpt')



show_reconstructed_digits(inputs, outputs, './model/my_model_stacked_denoising_dropout.ckpt')



########################################################################################################################
########################################################################################################################
