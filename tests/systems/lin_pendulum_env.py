# Copyright (c) 2023 Louis Jouret
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT


import gym
from gym import spaces
import polytope as pc
import pygame
import numpy as np


class invPendulum(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 25}

    def __init__(self, render_mode="human", max_steps=500, initState=[-4, 0, 0, 0], goal=[4, 0, 0, 0]):
        self.window_size = 510
        self.field_limit = [-5.1, 5.1]
        self.successThreshold = 0.5
        self.factor = self.window_size / \
            (self.field_limit[1] - self.field_limit[0])

        self.initState = initState
        self.time = 0
        self.dt = 0.05
        self.max_speed = 1.1
        self.maxTime = max_steps*self.dt
        self.timePenalty = 0
        self.actionPenalty = 0
        self.distancePenalty = 0  # -0.005
        self.obstaclePenalty = -10
        self.reward_goal = 1000

        self.obstacle_color = (100, 0, 0)
        self.goal_color = (0, 200, 100)
        self.pendulum_color = (0, 0, 0)
        self.background_color = (245, 240, 200)

        self.m = 0.05
        self.l = 3
        self.g = 9.81
        self.M = 0.01

        self.vertices1 = [self.pos_to_pixel([-2.75, 1 + self.l]),
                          self.pos_to_pixel([-1.25, 1 + self.l]),
                          self.pos_to_pixel([-1.25, 6]),
                          self.pos_to_pixel([-2.75, 6])]

        self.vertices2 = [self.pos_to_pixel([1.25, 1 + self.l]),
                          self.pos_to_pixel([2.75, 1 + self.l]),
                          self.pos_to_pixel([2.75, 6]),
                          self.pos_to_pixel([1.25, 6])]

        self.A = np.array([[1, self.dt, 0, 0],
                           [0, 1, self.m*self.dt*self.g/self.M, 0],
                           [0, 0, 1, self.dt],
                           [0, 0, self.dt*self.g*(self.m + self.M)/(self.M*self.l), 1]],
                          dtype=np.float32)

        self.B = np.array([[0], [self.dt/self.M], [0],
                          [self.dt/(self.M*self.l)]], dtype=np.float32)

        self.state = {
            'x': initState[0],
            'dx': initState[1],
            'phi': initState[2],
            'w': initState[3]
        }

        self.goal = {
            'x': goal[0],
            'dx': goal[1],
            'phi': goal[2],
            'w': goal[3]
        }

        self.max_steps = max_steps
        self.history = []

        self.actions = ["horizontal_force"]

        self.observations = ['x', 'dx', 'phi', 'w']

        low = np.array(-1, dtype=np.float32)
        high = np.array(1, dtype=np.float32)
        self.action_space = spaces.Box(
            low=low, high=high)

        self.first_run = True
        self.obstacle_set = []

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.observation_space = self.make_pendulum_obs_space()

        self.window = None
        self.clock = None

    def make_pendulum_obs_space(self):
        observations = ['x', 'dx', 'phi', 'w']

        lower_obs_bound = {
            'x': - np.inf,
            'dx': - np.inf,
            'phi': 0,
            'w': -np.inf
        }
        higher_obs_bound = {
            'x': np.inf,
            'dx': np.inf,
            'phi': 2*np.pi,
            'w': np.inf
        }

        low = np.array([lower_obs_bound[obs] for obs in observations])
        high = np.array([higher_obs_bound[obs] for obs in observations])
        shape = (len(observations),)
        return gym.spaces.Box(low, high, shape)

    def _get_obs(self):
        return np.array([self.state[obs] for obs in self.observations])

    def reset(self, initial_state=None):

        self.history = []

        if initial_state is None:
            self.state = {
                'x': np.random.uniform(-5, 5),
                'dx': 0,
                'phi': np.random.uniform(-0.7, 0.7),
                'w': 0
            }
        else:
            self.state = {
                'x': initial_state[0],
                'dx': initial_state[1],
                'phi': initial_state[2],
                'w': initial_state[3]
            }

        self.time = 0

        observation = self._get_obs()

        if self.render_mode == "human":
            self._render_frame()

        return observation

    def step(self, action):
        self.action = action
        state = self._get_obs()
        state = np.reshape(state, (-1, 1))
        self.history.append(state)
        observation = np.dot(self.A, state) + self.B * action
        observation = np.reshape(observation, (-1,))
        terminated = self.check_done()
        reward = self.get_reward()
        self.time += self.dt

        self.state = {
            'x': observation[0],
            'dx': observation[1],
            'phi': observation[2],
            'w': observation[3]
        }

        self.time += self.dt

        return observation, reward, terminated, {}

    def get_reward(self):
        actionAmplitude = np.abs(self.action[0])
        angle = np.sqrt((180*(self.state['phi'] - self.goal['phi'])/np.pi)**2)
        ball_x = self.state['x'] - self.l * np.sin(self.state['phi'])
        dist = (ball_x - self.goal['x'])**2
        speed = np.abs(self.state['w'])

        if self.check_done():
            if angle <= 3 and dist <= 0.1 and speed <= 0.1:
                return self.reward_goal
            elif self.time > self.maxTime:
                return self.distancePenalty*dist + self.actionPenalty * actionAmplitude
            else:
                return self.timePenalty + self.obstaclePenalty +\
                    self.distancePenalty*dist + self.actionPenalty * actionAmplitude
        else:
            return self.timePenalty + self.distancePenalty*dist + self.actionPenalty * actionAmplitude

    def check_done(self):
        ball_center = [self.state['x'] - self.l *
                       np.sin(self.state['phi']), 1 + self.l*np.cos(self.state['phi'])]
        angle = np.sqrt((180*(self.state['phi'] - self.goal['phi'])/np.pi)**2)
        dist = np.sqrt((self.state['x'] - self.goal['x'])**2)
        speed = np.abs(self.state['w'])

        if angle <= 3 and dist <= 0.1 and speed <= 0.1:
            return 1
        if self.time > self.maxTime:
            return 1
        elif np.abs(self.state['phi']) >= 0.7:
            return 1
        elif np.abs(self.state['x']) >= 5:
            return 1
        elif (-3 <= ball_center[0] <= -1 or 1 <= ball_center[0] <= 3) and ball_center[1] >= 3.8:
            return 1
        else:
            return 0

    def pos_to_pixel(self, pos):
        return (int(pos[0] * self.factor + self.window_size / 2),
                -int(pos[1] * self.factor - self.window_size / 2))

    def pixel_to_pos(self, pixel):
        return ((pixel[0] - self.window_size / 2) / self.factor,
                (-pixel[1] + self.window_size / 2) / self.factor)

    def get_vertices(self, polytope_num=None):

        a0 = -2.75
        a1 = -1.25
        delta_a = self.l*np.sqrt(1-0.93**2)
        theta_max = np.arccos(0.93)

        vertices = [(a0 + delta_a, theta_max),
                    (a0 - delta_a, -theta_max),
                    (a1 - delta_a, -theta_max),
                    (a1 + delta_a, theta_max),
                    (a0 + delta_a, theta_max),
                    (a0 - delta_a, -theta_max),
                    (a1 - delta_a, -theta_max),
                    (a1 + delta_a, theta_max)]

        return np.array(vertices)

    def render(self):
        if self.render_mode == "human":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            pygame.display.set_caption('Inverted Pendulum')
            self.window = pygame.display.set_mode(
                (self.window_size, int(self.window_size/2)))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        self.canvas = canvas
        canvas.fill(self.background_color)

        cart_center = self.pos_to_pixel([self.state['x'], 1])
        ball_center = self.pos_to_pixel(
            [self.state['x'] - self.l*np.sin(self.state['phi']), 1 + self.l*np.cos(self.state['phi'])])
        min_height = self.pos_to_pixel([-5, 3.2])

        # draw guideline
        pygame.draw.line(
            canvas,
            self.pendulum_color,
            self.pos_to_pixel([-5, 1]),
            self.pos_to_pixel([5, 1])
        )
        # draw obstacles

        pygame.draw.rect(
            canvas,
            self.obstacle_color,
            pygame.Rect(
                0,
                min_height[1],
                self.factor*0.1,
                self.window_size))

        pygame.draw.rect(
            canvas,
            self.obstacle_color,
            pygame.Rect(
                self.window_size - self.factor*0.1,
                min_height[1],
                self.factor*0.1,
                self.window_size))

        pygame.draw.rect(
            canvas,
            self.obstacle_color,
            pygame.Rect(
                min_height[0],
                min_height[1],
                self.factor*10,
                self.factor*0.1
            ))

        pygame.draw.polygon(
            canvas,
            self.obstacle_color,
            self.vertices1)

        pygame.draw.polygon(
            canvas,
            self.obstacle_color,
            self.vertices2)

        # draw goal
        pygame.draw.circle(
            canvas,
            self.goal_color,
            self.pos_to_pixel([4, 1 + self.l]),
            self.factor*0.3)

        # draw state

        pygame.draw.rect(
            canvas,
            self.pendulum_color,
            pygame.Rect(
                cart_center[0] - self.factor*0.5,
                cart_center[1] - self.factor*0.3,
                self.factor,
                self.factor*0.6
            ))

        pygame.draw.line(
            canvas,
            self.pendulum_color,
            cart_center,
            ball_center
        )

        pygame.draw.circle(
            canvas,
            self.pendulum_color,
            ball_center,
            self.factor*0.2)

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

    def rotation_matrix(self, angle):
        return np.array([[np.cos(angle), -np.sin(angle)],
                         [np.sin(angle), np.cos(angle)]])

    def expand_safety_polytope(self, vertices):
        new_vertices = []
        vertices = np.array(vertices)
        polytope = pc.qhull(vertices)
        exp_polytope = pc.qhull(np.array([[0, 0], [5, 0], [5, 5], [0, 5]]))

        dist = self.dt*self.max_speed
        for vertex in vertices:
            actif = []
            for normal, b in zip(polytope.A, polytope.b):
                constraint = pc.Polytope(
                    np.array([normal]), np.array([[b]]))
                normal = normal / np.linalg.norm(normal)
                if self.is_on_boundary_point(constraint, vertex):
                    actif.append(normal)
                    exp_vertex = vertex + dist * normal/np.linalg.norm(normal)
                    exp_b = np.dot(normal, exp_vertex)
                    exp_constraint = pc.Polytope(
                        np.array([normal]), np.array([[exp_b]]))
                    exp_polytope = pc.Polytope(np.vstack((exp_polytope.A, exp_constraint.A)),
                                               np.hstack((exp_polytope.b, exp_constraint.b)))
                    exp_polytope = pc.reduce(exp_polytope)

            avg_normal = np.mean(actif, axis=0)
            cut_vertex = vertex + dist * avg_normal/np.linalg.norm(avg_normal)
            cut_b = np.dot(avg_normal, cut_vertex)
            cut_constraint = pc.Polytope(
                np.array([avg_normal]), np.array([[cut_b]]))
            exp_polytope = pc.Polytope(np.vstack((exp_polytope.A, cut_constraint.A)),
                                       np.hstack((exp_polytope.b, cut_constraint.b)))
            exp_polytope = pc.reduce(exp_polytope)

        new_vertices = pc.extreme(exp_polytope)
        centroid = np.mean(new_vertices, axis=0)
        new_vertices = sorted(new_vertices, key=lambda x: np.arctan2(
            x[1] - centroid[1], x[0] - centroid[0]))
        return np.array(new_vertices)

    def normal(self, obstacles, vertex):
        normals = []
        for obstacle in obstacles:
            for normal, b in zip(obstacle.A, obstacle.b):
                constraint = pc.Polytope(
                    np.array([normal]), np.array([[b]]))
                normal = normal / np.linalg.norm(normal)
                if self.is_on_boundary_point(constraint, vertex):
                    normals.append(normal)

        normal = np.mean(normals, axis=0)
        normal = normal / np.linalg.norm(normal)

        return normal

    def is_on_boundary_point(self, polytope, x):
        tol = 1e-8
        for A, b in zip(polytope.A, polytope.b):
            e = np.dot(A, x) - b
            if np.any(abs(e) < tol):
                return True
        return False

    def get_obstacle_set(self):
        """ get the set of pixels which are colored by the obstacle method"""
        obstacle_set = []
        for pixel_x in range(self.window_size):
            for pixel_y in range(self.window_size):
                if self.canvas.get_at((pixel_x, pixel_y)) == self.obstacle_color:
                    obstacle_set.append((pixel_x, pixel_y))
        return obstacle_set
