# Copyright (c) 2023 louisjouret
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import numpy as np
from ddpg_utils.agent import Agent
from systems.mobile_robot_env import MobileRobot

env = MobileRobot()
agent = Agent(len(env.actions), len(env.observations))
# run the model with a dummy input to initialize the weights
dummy_state = np.zeros((1, len(env.observations)))
agent.actorMain(dummy_state)
agent.actorMain.summary(line_length=100)

episodes = 50000
movAvglength = 100
episodeScore = []
episodeAvgScore = []
lastAvg = 0
best = -100
history_succes = []
percent_succes = 0
env.render()
episodespan = 0

best = -100

for episode in range(episodes):
    done = False
    score = 0
    observation = env.reset()
    while not done:
        if episode % 10 == 0:
            env.render()
        action = agent.act(np.array([observation]))
        nextObservation, reward, done, _ = env.step(action[0])
        agent.replayBuffer.storexp(
            observation, action, reward, done, nextObservation)
        agent.train()
        observation = nextObservation
        score += reward
    if reward > 0:
        history_succes.append(1)
    else:
        history_succes.append(0)
    episodeScore.append(score)
    if episode > movAvglength:
        avg = np.mean(episodeScore[-movAvglength:])
        percent_succes = np.sum(
            history_succes[-movAvglength:])*100/movAvglength
    else:
        avg = np.mean(episodeScore)
        percent_succes = np.sum(
            history_succes)*100/movAvglength
    print(
        f" episode {episode} has a score of {score}. The average score over 100 episode is {round(avg,2)}.")
    episodeAvgScore.append(avg)

    if avg > best:
        agent.actorMain.save_model()
        agent.criticMain.save_model()
        best = avg
