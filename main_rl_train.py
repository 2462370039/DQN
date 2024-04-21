import random

import gym
import numpy as np
import torch
from torch import nn

import agent

env = gym.make('CartPole-v0')
s = env.reset()

EPSILON_DECAY = 10000
EPSILON_START = 1.0
EPSILON_END = 0.02

n_episodes = 5000
n_time_steps = 500

n_state = len(s)
n_action = env.action_space.n

agent = agent.Agent(n_state, n_action)

REWARDS_BUFFER = []
for episode_i in range(n_episodes):
    episode_reward = 0
    for step_i in range(n_time_steps):
        #------------------
        # With probability epsilon, select a random action a_t
        #------------------
        epsilon = np.interp(episode_i*n_time_steps+step_i, [0, EPSILON_DECAY], [EPSILON_START, EPSILON_END])
        random_sample = random.random()

        if random_sample < epsilon:
            a = env.action_space.sample()
        else:
            #------------------
            # Otherwise, select a_t = argmax_a Q(s_t, a)
            #------------------
            a = agent.online_net.act(s)

        #------------------
        # Execute action a_t in emulator and observe reward r_t and image x_{t+1}
        # Set s_{t+1} = s_t, a_t, x_{t+1} and preprocess φ_{t+1} = φ(s_{t+1})
        # Store transition (φ_t, a_t, r_t, φ_{t+1}) in D
        #------------------
        s_, r, done, _ = env.step(a)
        agent.memo.add_memo(s, a, r, s_, done)
        s = s_
        episode_reward += r

        if done:
            s = env.reset()
            REWARDS_BUFFER.append(episode_reward)
            break

        if np.mean(np.mean(REWARDS_BUFFER[:episode_i])) >= 100:
            while True:
                a = agent.online_net.act(s)
                s, r, done, _ = env.step(a)
                env.render()

                if done:
                    env.reset()
        #------------------
        # Sample random minibatch of transitions (φ_j, a_j, r_j, φ_{j+1}) from D
        #------------------
        batch_s, batch_a, batch_r, batch_done, batch_s_ = agent.memo.sample()

        #------------------
        # Set y_j = r_j for terminal φ_{j+1}
        # Set y_j = r_j + gamma max_a Q_target(φ_{j+1}, a; θ_target) for non-terminal φ_{j+1}
        #------------------
        # Compute Q-learning target
        target_q_values = agent.target_net(batch_s_)
        max_target_q_values = target_q_values.max(dim=1, keepdim=True)[0]
        targets = batch_r + agent.GAMMA * max_target_q_values * (1 - batch_done)

        #------------------
        # Perform a gradient descent step on (y_j - Q(φ_j, a_j; θ))^2 with respect to the network parameters θ
        #------------------
        # Compute q-values
        q_values = agent.online_net(batch_s)
        a_q_values = torch.gather(input=q_values, dim=1, index=batch_a)

        # Compute loss
        loss = nn.functional.smooth_l1_loss(a_q_values, targets)

        # Optimize (Gradient Descent)
        agent.optimizer.zero_grad()
        loss.backward()
        agent.optimizer.step()

    #------------------
    # Every C steps reset Q_target = Q
    #------------------
    # Update target network
    if episode_i % 10 == 0:
        agent.target_net.load_state_dict(agent.online_net.state_dict())

        # Print average reward
        print(f'Episode {episode_i} Avg Reward: {np.mean(REWARDS_BUFFER[:episode_i])}')
