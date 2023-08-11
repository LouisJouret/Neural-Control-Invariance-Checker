# Copyright (c) 2023 Louis Jouret
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT


from systems.mass_spring_damper_env import MassSpringDamper
from systems.mobile_robot_env import MobileRobot
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import polytope as pc
import nn_invariance_check as nci

####### Dynamic System #######
# The should be characterized by two matrices A and B such that x_next = Ax + B.

env = MobileRobot()
### Uncomment for MassSpringDamper ###
# n = 1
# env = MassSpringDamper(nb_masses=n)
######################################
A = env.A
B = env.B

####### FNN Controller #######
# The controller should be loaded from a Tensorflow SavedModel file.
nn_controller = tf.keras.models.load_model("tests/nn_controller")

####### Verification set #######
# The set that will be verified should be expressed as S\O where S is a convex polytope and
# O is the union of all convex polytopic obstacles. O should be expressed as a list of polytopes.

### Uncomment for MassSpringDamper ###
# S = invariance_approach.expand_polytope_to_2D(
#     pc.qhull(np.array([[0, 0], [1, 1], [1, -1]])), n)
# O = None
######################################

S = pc.qhull(np.array([[-5, 5], [-5, -5], [5, -5], [5, 5]]))
O = [pc.qhull(np.array(env.get_obstacle_vertices(1))),
     pc.qhull(np.array(env.get_obstacle_vertices(3)))]


####### Check Controller Safety #######
regions, good_vertices, bad_vertices = nci.check_controller_safety(
    S, O, nn_controller, A, B)

##### Visualization #####
# If the system has only 2 dimensions, the vertices and linear regions can be visualized as such:
if A.shape[1] == 2:
    fig, ax = plt.subplots(figsize=(8, 8))
    for region in regions:
        region.plot(ax, linewidth=0.1)
    if O is not None:
        for obstacle in O:
            obstacle.plot(ax, alpha=0.3, linewidth=0.1, color='black')
    for gv in good_vertices:
        plt.scatter(gv[0], gv[1], c='g', s=2)
    for bv in bad_vertices:
        plt.scatter(bv[0], bv[1], c='r', s=2)
    plt.show()
