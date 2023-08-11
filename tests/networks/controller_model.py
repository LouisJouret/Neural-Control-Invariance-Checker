# Copyright (c) 2023 Louis Jouret
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import tensorflow as tf


class NNController(tf.keras.Model):
    def __init__(self, stateDim, actionDim, layer1Dim=512, layer2Dim=512):
        super().__init__()
        self.stateDim = stateDim
        self.actionDim = actionDim
        self.layer1Dim = layer1Dim
        self.layer2Dim = layer2Dim
        self.createModel()

    def createModel(self):
        "creates a tensorflow model of 2 leaky-relu layers followed by a custom piece-wise linear tanh output layer"
        self.l1 = tf.keras.layers.Dense(
            input_shape=self.stateDim, units=self.layer1Dim, activation=tf.keras.layers.LeakyReLU(
                alpha=0.01),
            kernel_regularizer=tf.keras.regularizers.l2(0.01), bias_regularizer=tf.keras.regularizers.l2(0.01))
        self.l2 = tf.keras.layers.Dense(
            units=self.layer2Dim, activation=tf.keras.layers.LeakyReLU(
                alpha=0.01),
            kernel_regularizer=tf.keras.regularizers.l2(0.01), bias_regularizer=tf.keras.regularizers.l2(0.01))
        self.lact = PLULayer(self.actionDim)

    def call(self, state):
        x = self.l1(state)
        x = self.l2(x)
        x = self.lact(x)
        return x

    def save_model(self):
        print("\n saving model ...")
        self.save("tests/nn_controller")


class PLULayer(tf.keras.layers.Layer):
    def __init__(self, num_outputs=2):
        super().__init__()
        self.num_outputs = num_outputs

    def build(self, input_shape):
        self.kernel = self.add_weight("kernel",
                                      shape=[int(input_shape[-1]),
                                             self.num_outputs])
        self.bias = self.add_weight("bias",
                                    shape=[self.num_outputs],
                                    initializer="zeros")

    def call(self, x):
        x = tf.matmul(x, self.kernel) + self.bias
        return pieceWiseLinear(x)


@ tf.custom_gradient
def pieceWiseLinear(x):
    condition1 = tf.math.greater(tf.abs(x), 100)

    condition2 = tf.math.greater(tf.abs(x), 1.5)
    condition3 = tf.math.less_equal(tf.abs(x), 100)
    condition4 = tf.math.logical_and(condition2, condition3)

    condition5 = tf.math.greater(tf.abs(x), 0.5)
    condition6 = tf.math.less_equal(tf.abs(x), 1.5)
    condition7 = tf.math.logical_and(condition5, condition6)
    x = tf.where(condition1, tf.sign(x)*1.099, x)
    x = tf.where(condition4, 0.001*x + tf.sign(x)*0.999, x)
    x = tf.where(condition7, 0.5*x + tf.sign(x)*0.25, x)

    def grad(upstream):
        dy_dx = tf.ones_like(x)
        dy_dx = tf.where(condition1, 0.0, dy_dx)
        dy_dx = tf.where(condition4, 0.001, dy_dx)
        dy_dx = tf.where(condition7, 0.5, dy_dx)

        return dy_dx * upstream

    return x, grad
