# Copyright (c) 2023 Louis Jouret
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import tensorflow as tf


class Critic(tf.keras.Model):
    def __init__(self, stateDim, actionDim, layer1Dim=512, layer2Dim=512):
        super().__init__()
        self.stateDim = stateDim
        self.actionDim = actionDim
        self.layer1Dim = layer1Dim
        self.layer2Dim = layer2Dim
        self.createModel()

    def createModel(self):
        "creates a tensorflow model of 2 leaky relu layers followed by a linear output"
        self.l1 = tf.keras.layers.Dense(
            input_shape=self.stateDim, units=self.layer1Dim, activation=tf.keras.layers.LeakyReLU(
                alpha=0.01),
            kernel_regularizer=tf.keras.regularizers.l2(0.01), bias_regularizer=tf.keras.regularizers.l2(0.01))
        self.l2 = tf.keras.layers.Dense(
            units=self.layer2Dim, activation=tf.keras.layers.LeakyReLU(
                alpha=0.01),
            kernel_regularizer=tf.keras.regularizers.l2(0.01), bias_regularizer=tf.keras.regularizers.l2(0.01))
        self.lq = tf.keras.layers.Dense(1, activation=None)

    def call(self, state, action):
        x = self.l1(tf.concat([state, action], axis=1))
        x = self.l2(x)
        x = self.lq(x)
        return x

    def save_model(self):
        print("saving critic model ...")
        self.save("Ddpg/critic")
