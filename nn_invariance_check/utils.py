# Copyright (c) 2023 Louis Jouret
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT


from __future__ import annotations
import tensorflow as tf
import polytope as pc
import numpy as np
import cdd


def compute_gradient_approach(polytope, nn_controller, weights, nb_layer):
    """ Compute the active parameters w_active and b_active of a polytope """

    W = np.transpose(weights[0])
    B = np.transpose([weights[1]])
    cheby_center = pc.cheby_ball(polytope)[1]
    b_comp, grad = gradient(nn_controller, cheby_center, nb_layer)
    w_active = np.matmul(W, grad)
    b_active = np.matmul(W, b_comp) + B
    return w_active, b_active


def gradient(nn_controller, x, nb_layer):
    """ Computes the gradient of the neural network output with respect to the input """

    if nb_layer == 0:
        return np.zeros((x.shape[0], 1)), np.identity(x.shape[0])

    x_tf = tf.Variable([x], dtype=tf.float32)

    hidden_output, gradients = _comp_jacobian(x_tf, nn_controller, nb_layer)
    grad = gradients.numpy()[0, :, 0, :]
    hidden_output = hidden_output.numpy().reshape((-1, 1))
    b_comp = hidden_output - np.matmul(grad, np.transpose(x)).reshape((-1, 1))

    return b_comp, grad


@tf.function
def _comp_jacobian(x, nn_controller, nb_layer):
    """ Computes the jacobian matrix of the output of a layer with respect to the models input """

    with tf.GradientTape() as tape:
        hidden_output = x
        for i, layer in enumerate(nn_controller.layers):
            hidden_output = layer(hidden_output)
            if i == nb_layer-1:
                break
    return hidden_output, tape.jacobian(hidden_output, x)


def is_limitroph(polytope, input_space, obstacles):
    """ Checks if a polytope intersects the border of the input space or an obstacle """

    diff = pc.mldivide(polytope, input_space)
    if pc.is_fulldim(diff):
        return True
    if obstacles is not None:
        for obstacle in obstacles:
            inter = pc.intersect(polytope, obstacle)
            diff = pc.mldivide(polytope, obstacle)
            if pc.is_fulldim(inter) and pc.is_fulldim(diff):
                return True
    return False


def expand_polytope_to_2D(polytope, n):
    m, l = polytope.A.shape
    A = np.zeros((n * m, n * l))
    B = np.zeros((n * m, 1))
    for j in range(n):
        A[j * m:(j + 1) * m, j * l:(j + 1) * l] = polytope.A
        B[j * m:(j + 1) * m, 0] = polytope.b

    return pc.Polytope(A, B)


def cut(polytope, w_active, b_active, input_space, obstacles, activation):
    """ Division of a polytope when cut by the active parameters w_active and b_active """

    kids = [polytope]
    if activation == 'leaky_re_lu':
        breakpoints = [0]
    else:
        breakpoints = [-100, -1.5, -0.5, 0, 0.5, 1.5, 100]
    cuts = []
    for neuron in range(w_active.shape[0]):
        for bp in breakpoints:
            w = np.array(w_active[neuron, :])
            b = np.array(b_active[neuron])
            cut = pc.Polytope(np.vstack((polytope.A, -w)),
                              np.append(polytope.b, b - bp))
            cuts.append(cut)
    for cut in cuts:
        copy = list(kids)
        kids = []
        for region in copy:
            cutted = False
            kid1 = pc.intersect(region, cut)
            if pc.is_fulldim(kid1):
                kid2 = region.diff(kid1)
                if pc.is_fulldim(kid2):
                    if is_limitroph(kid1, input_space, obstacles):
                        kids.append(kid1)
                        cutted = True
                    if is_limitroph(kid2, input_space, obstacles):
                        kids.append(kid2)
                        cutted = True
                    cutted = True
            if not cutted:
                kids.append(region)
    return kids


def scale_polytope(polytope, point, scale_factor):
    """ Scale a polytope with respect to a point """

    A = polytope.A
    b = polytope.b
    distances = np.matmul(A, np.array(point)) - b
    b += (1-scale_factor) * distances
    return pc.Polytope(A, b)


def linear_regions(nn_controller, input_space, obstacles):
    """ Compute the linear regions of the Neural Network """

    print("Generating the linear regions of the action space :")
    regions = [input_space]
    reduced_input_space = input_space.copy()
    reduced_input_space = scale_polytope(
        reduced_input_space, pc.cheby_ball(input_space)[1], 0.999)
    for layer_idx, layer in enumerate(nn_controller.layers):
        if layer_idx < len(nn_controller.layers) - 1:
            activation = 'leaky_re_lu'
        else:
            activation = 'tanh'
        weights = layer.weights
        new_regions = []
        l = len(regions)
        for idx, region in enumerate(regions):
            if layer_idx != 0:
                print(
                    f"* Layer {layer_idx + 1} cutting ... {round(100*idx/l)} %", end="\r")
            else:
                # print(f"First layer cutting ...", end="\r")
                print(f"* First layer cutting: ...", end="\r")
            w_active, b_active = compute_gradient_approach(
                region, nn_controller, weights, layer_idx)
            kids = cut(region, w_active, b_active,
                       reduced_input_space, obstacles, activation)
            new_regions.extend(kids)
        if layer_idx != 0:
            print(
                f"* Up to layer {layer_idx + 1}, {len(new_regions)} regions were created.")
        else:
            print(f"* First layer created {len(new_regions)} regions.")
        regions = new_regions
    return new_regions


def intersect_boundary_point(polytope, obstacle):
    """ Returns the vertices of the intersection between a polytope and an obstacle """
    intersection_points = []
    intersection = pc.intersect(polytope, obstacle)
    if pc.is_fulldim(intersection) > 0:
        mat = cdd.Matrix(
            np.hstack((np.transpose([intersection.b]), -intersection.A)))
        poly = cdd.Polyhedron(mat)
        v = np.array([list(x) for x in poly.get_generators()])
        vertices = v[:, 1:] / v[:, [0]]
        for vertex in vertices:
            is_on_bound, _ = point_is_on_boundary(obstacle, vertex)
            if is_on_bound:
                intersection_points.append(vertex)
    if len(intersection_points) == 0:
        intersection_points = None
    return intersection_points


def point_is_on_boundary(polytope, x):
    """ Checks if a point is on the boundary of a polytope """

    tol = 1e-8
    nb_actif_constraints = 0
    for A, b in zip(polytope.A, polytope.b):
        e = np.dot(A, x) - b
        if np.any(abs(e) < tol):
            nb_actif_constraints += 1
    if nb_actif_constraints == 0:
        return False, None
    else:
        return True, nb_actif_constraints


def intersection_points(limit_regions, obstacle):
    """ Compute all vertices that need to be checked """

    intersection_points = []
    for idx, polytope in enumerate(limit_regions):
        print(
            f"Generating vertices: {round(100*idx/len(limit_regions))} %", end="\r")
        points = intersect_boundary_point(polytope, obstacle)
        if points is not None:
            intersection_points.extend(points)
    print(
        f"Generating vertices: done!", end="\n")
    # remove doublons
    intersection_points = list(
        set([tuple(point) for point in intersection_points]))
    if len(intersection_points) == 0:
        intersection_points = None
    return intersection_points


def check_vertex(limit_regions, input_space, obstacles, nn_controller, A, b):
    """ Check if the velocity vector points back in the safe set """
    good_points = []
    bad_points = []

    points = intersection_points(limit_regions, input_space)
    print(f"{len(points)} points need to be verified for the input space: ...", end="\r")
    if points is not None:
        dataset = tf.data.Dataset.from_tensor_slices(np.array(points))
        dataset = dataset.batch(len(points))  # Batch the entire dataset

        # Iterate through the dataset and perform the feedforward pass
        for batch_inputs in dataset:
            predictions = nn_controller(batch_inputs)
        actions = predictions.numpy()
        next_points = np.array(A @ np.array(points).T + b @ actions.T).T
        for pt, npt in zip(points, next_points):
            if pc.is_inside(input_space, npt):
                good_points.append(pt)
            else:
                bad_points.append(pt)
    print(f"{len(points)} points need to be verified for the input space: done!", end="\n")
    if obstacles is not None:
        for idx, obstacle in enumerate(obstacles):
            points = intersection_points(limit_regions, obstacle)
            print(
                f"{len(points)} points need to be verified for obstacle {idx + 1}: ...", end="\r")
            if points is not None:
                actions = nn_controller(points).numpy()
                next_points = np.array(
                    A @ np.array(points).T + b @ actions.T).T
                for pt, npt in zip(points, next_points):
                    _, nb_actif_const = point_is_on_boundary(obstacle, pt)
                    dist = obstacle.A @ np.array(npt) - obstacle.b
                    if np.sum(dist > 0, axis=0) == nb_actif_const:
                        good_points.append(pt)
                    else:
                        bad_points.append(pt)
            print(
                f"{len(points)} points need to be verified for obstacle {idx + 1}: done!", end="\n")
    return good_points, bad_points


def check_controller_safety(input_space, obstacles, nn_controller, A, b):
    """ 
    Check the controlled invariance of the set input_space\obstacels when 
    the system x_next = Ax + b is controlled by a neural network controller
    """

    limit_regions = linear_regions(nn_controller, input_space, obstacles)
    good_points, bad_points = check_vertex(
        limit_regions, input_space, obstacles, nn_controller, A, b)
    if len(bad_points) > 0:
        print(
            f"The neural network controller is not safe!")

    return limit_regions, good_points, bad_points
