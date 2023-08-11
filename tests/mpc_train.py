# Copyright (c) 2023 Louis Jouret
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from networks.controller_model import NNController
from systems.mass_spring_damper_env import MassSpringDamper
import do_mpc
from casadi import *


@tf.function
def train(nn_controller, x, u_hat):
    u_hat = tf.convert_to_tensor([u_hat], dtype=tf.float32)
    x = tf.convert_to_tensor([x], dtype=tf.float32)
    with tf.GradientTape() as tape:
        u = nn_controller(x)
        loss = tf.reduce_mean(tf.square(u - u_hat))

    grad = tape.gradient(loss, nn_controller.trainable_variables)
    nnOptimizer.apply_gradients(zip(grad, nn_controller.trainable_variables))
    return loss


####### Number of masses #######
n = 1
################################

env = MassSpringDamper(nb_masses=n)
A = env.A
B = env.B
episodes = 10000

nn_controller = NNController((2*n,), n, 64, 64)
nnOptimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
nn_controller.compile(optimizer=nnOptimizer)
dummy_state = np.ones((1, 2*n), dtype=np.float32)
nn_controller(dummy_state)
nn_controller.summary()

avg_loss = []
done = 0

##################
###### MPC #######
##################

model_type = 'discrete'
model = do_mpc.model.Model(model_type)

_x = model.set_variable(var_type='_x', var_name='x', shape=(2*n, 1))
_u = model.set_variable(var_type='_u', var_name='u', shape=(n, 1))

x_next = A@_x + B@_u

model.set_rhs('x', x_next)
model.set_expression(expr_name='cost', expr=sum1(_x**2))
model.setup()
mpc = do_mpc.controller.MPC(model)
surpress_ipopt = {'ipopt.print_level': 0, 'ipopt.sb': 'yes', 'print_time': 0}
setup_mpc = {
    'n_robust': 0,
    'n_horizon': 7,
    't_step': env.dt,
    'state_discretization': 'discrete',
    'store_full_solution': True,
    'nlpsol_opts': surpress_ipopt,
}

mpc.set_param(**setup_mpc)
mterm = model.aux['cost']  # terminal cost
lterm = model.aux['cost']  # terminal cost
# stage cost

mpc.set_objective(mterm=mterm, lterm=lterm)

mpc.set_rterm(u=1e-4)  # input penalty
u_max = np.ones((n, 1))

# lower bounds of the states
mpc.bounds['lower', '_x', 'x'] = np.array([-1] * 2*n)

# upper bounds of the states
mpc.bounds['upper', '_x', 'x'] = np.array([1] * 2 * n)

# lower bounds of the input
mpc.bounds['lower', '_u', 'u'] = -u_max

# upper bounds of the input
mpc.bounds['upper', '_u', 'u'] = u_max
mpc.setup()

estimator = do_mpc.estimator.StateFeedback(model)
simulator = do_mpc.simulator.Simulator(model)
simulator.set_param(t_step=env.dt)
simulator.setup()

# Initial state
x = np.random.uniform(-1, 1, (n, 1))
v = np.random.uniform(-1, 1, (n, 1))
x0 = np.vstack((x, v))
mpc.x0 = x0
simulator.x0 = x0
estimator.x0 = x0

# Use initial state to set the initial guess.
mpc.set_initial_guess()


######################
######## Train #######
######################

avg_loss = []

for episode in range(episodes):
    env.reset()
    u_hat = np.zeros(n)
    done = 0
    while not done:
        u_hat = np.array(mpc.make_step(env.states)).flatten()
        states, done = env.step(u_hat)

        loss = train(nn_controller, states, u_hat)
        avg_loss.append(float(loss))

        if len(avg_loss) > 1000:
            avg_loss.pop(0)  # Remove the oldest value from the rolling window

        # Calculate the moving average
        moving_average = sum(avg_loss) / len(avg_loss)
        print('episode = ', episode, 'loss = ',
              moving_average, end='\r')

    if episode % 50 == 0:
        nn_controller.save_model()
