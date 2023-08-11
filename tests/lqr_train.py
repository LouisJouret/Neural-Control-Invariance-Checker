# Copyright (c) 2023 Louis Jouret
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT


import control as ct
import numpy as np
from networks.controller_model import NNController
from systems.mass_spring_damper_env import MassSpringDamper
import tensorflow as tf


@tf.function
def train(actor, x, u_hat):
    u_hat = tf.convert_to_tensor([u_hat], dtype=tf.float32)
    x = tf.convert_to_tensor([x], dtype=tf.float32)
    with tf.GradientTape() as tape:
        u = actor(x)
        loss = tf.reduce_mean(tf.square(u - u_hat))

    grad = tape.gradient(loss, actor.trainable_variables)
    nnOptimizer.apply_gradients(zip(grad, actor.trainable_variables))
    return loss


####### Number of masses #######
n = 1
################################

env = MassSpringDamper(nb_masses=n)
A = env.A
B = env.B
Q = np.eye(2*n)
R = np.eye(n)
K, S, E = ct.dlqr(A, B, Q, R)
episodes = 10000

nn_controller = NNController((2*n,), n, 8, 8)
nnOptimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
nn_controller.compile(optimizer=nnOptimizer)
dummy_state = np.ones((1, 2*n), dtype=np.float32)
nn_controller(dummy_state)
nn_controller.summary()

avg_loss = []
done = 0

for episode in range(episodes):
    env.reset()
    u_hat = np.zeros(n)
    done = 0
    while not done:
        u_hat = -np.matmul(K, env.states)
        u_hat = np.clip(u_hat, -1, 1)
        states, done = env.step(u_hat)

        loss = train(nn_controller, states, u_hat)
        avg_loss.append(float(loss))

        if len(avg_loss) > 2000:
            avg_loss.pop(0)  # Remove the oldest value from the rolling window

        # Calculate the moving average
        moving_average = sum(avg_loss) / len(avg_loss)
        print('episode = ', episode, 'loss = ',
              moving_average, end='\r')

    if episode % 50 == 0:
        nn_controller.save_model()
