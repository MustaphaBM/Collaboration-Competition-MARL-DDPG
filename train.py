from collections import deque

import numpy as np
import torch
import yaml

from ddpg_agent import Agent


def train_ddpg(
    env,
    brain,
    agent_config,
    n_episodes=5000,
):
    """Training function

    Params
    ======
        env : environment
        brain (str) : brain's name
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        time_steps (int) :
    """
    scores_deque = deque(maxlen=100)
    scores = []
    avg_score_list = []
    agent = Agent(**agent_config)

    for i_episode in range(1, n_episodes + 1):
        env_info = env.reset(train_mode=True)[brain]
        states = env_info.vector_observations
        agent.reset()
        score = np.zeros(len(env_info.agents))
        while True:
            actions = agent.act(states)
            env_info = env.step(actions)[brain]
            next_states = env_info.vector_observations
            rewards = env_info.rewards
            dones = env_info.local_done
            agent.step(states, actions, rewards, next_states, dones)
            score += rewards
            states = next_states
            if np.any(dones):
                break

        scores_deque.append(np.mean(score))
        scores.append(np.mean(score))
        avg_score = np.mean(scores_deque)
        avg_score_list.append(avg_score)

        if i_episode % 100 == 0:
            print(
                "\rEpisode {}\tAverage Score: {:.5f}\tScore: {:.5f}".format(
                    i_episode, avg_score, np.mean(score)
                ),
                end="\n",
            )

        if np.mean(scores_deque) > 0.5:
            print(f"Enviroment solved in episode={i_episode} avg_score={avg_score:.5f}")

            torch.save(agent.actor_local.state_dict(), "checkpoint_actor.pth")
            torch.save(agent.critic_local.state_dict(), "checkpoint_critic.pth")

            break

    return scores, avg_score_list
