# Copyright (c) 2023 Louis Jouret
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import gym
from gym import spaces
import polytope as pc
import pygame
import numpy as np
import control as ct


class MassSpringDamper():

    def __init__(self, max_steps=2000, nb_masses=1):

        if nb_masses > 4 or nb_masses < 1:
            raise NotImplementedError(
                "The number of masses is limited to 4. Choose between 1 and 4.")

        self.initState = np.random.uniform(-1, 1, 2*nb_masses)
        self.goal = np.zeros((2*nb_masses))
        self.time = 0
        self.dt = 0.1
        self.maxTime = max_steps*self.dt

        k1 = 1
        k2 = 3
        k3 = 2
        k4 = 1
        K4 = np.array([[k1+k2, -k2, 0, 0], [-k2, k2+k3, -k3, 0],
                       [0, -k3, k3+k4, -k4], [0, 0, -k4, k4]])
        K3 = np.array([[k1+k2, -k2, 0], [-k2, k2+k3, -k3], [0, -k3, k3]])
        K2 = np.array([[k1+k2, -k2], [-k2, k2]])
        K1 = np.array([k1])

        c1 = 0.1
        c2 = 0.1
        c3 = 0.1
        c4 = 0.1
        C4 = np.array([[c1+c2, -c2, 0, 0], [-c2, c2+c3, -c3, 0],
                       [0, -c3, c3+c4, -c4], [0, 0, -c4, c4]])
        C3 = np.array([[c1+c2, -c2, 0], [-c2, c2+c3, -c3], [0, -c3, c3]])
        C2 = np.array([[c1+c2, -c2], [-c2, c2]])
        C1 = np.array([c1])

        m1 = 1
        m2 = 3
        m3 = 1
        m4 = 2
        M1 = np.diag([m1])
        M2 = np.diag([m1, m2])
        M3 = np.diag([m1, m2, m3])
        M4 = np.diag([m1, m2, m3, m4])

        M = [M1, M2, M3, M4]
        C = [C1, C2, C3, C4]
        K = [K1, K2, K3, K4]
        MC = np.matmul(np.linalg.inv(M[nb_masses-1]), C[nb_masses-1])
        MK = np.matmul(np.linalg.inv(M[nb_masses-1]), K[nb_masses-1])

        A_high = np.hstack(
            (np.zeros((nb_masses, nb_masses)), np.eye(nb_masses)))
        A_low = np.hstack((-MK, -MC))
        A_cont = np.vstack((A_high, A_low))
        B_cont = np.vstack((np.zeros((nb_masses, nb_masses)),
                           np.linalg.inv(M[nb_masses-1])))
        sys = ct.ss(A_cont, B_cont, np.eye(2*nb_masses),
                    np.zeros((2*nb_masses, nb_masses)))
        sysd = ct.c2d(sys, self.dt)
        self.A = sysd.A
        self.B = sysd.B

        self.max_steps = max_steps
        self.history = []

        self.actions = []
        self.states = self.initState

        action_low = np.array(-1, dtype=np.float32)
        action_high = np.array(1, dtype=np.float32)
        state_low = np.array(-2, dtype=np.float32)
        state_high = np.array(2, dtype=np.float32)
        self.action_space = spaces.Box(
            low=action_low, high=action_high)
        self.state_space = spaces.Box(
            low=state_low, high=state_high)

    def reset(self):
        self.history = []
        for idx in range(len(self.states)):
            self.states[idx] = np.random.uniform(-1, 1, (1,))
        self.time = 0

    def step(self, action):
        self.action = action
        self.history.append(self.states)
        observation = self.A @ np.squeeze(self.states) +\
            self.B @ action
        terminated = self.check_done()
        self.time += self.dt

        self.states = observation

        self.time += self.dt

        return observation,  terminated

    def check_done(self):
        if np.any(np.abs(self.states > 2)):
            return 1
        elif self.time > self.maxTime:
            return 1
        elif np.all(np.abs(self.states - self.goal) < 0.01):
            return 1
        else:
            return 0
