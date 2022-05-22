import numpy as np
import yaml
from matplotlib import pyplot as plt
from unityagents import UnityEnvironment

from train import train_ddpg

if __name__ == "__main__":

    env = UnityEnvironment(
        "/home/mustapha/Desktop/udacity_nano_degree/deep-reinforcement-learning/p3_collab-compet/Tennis_Linux/Tennis.x86_64"
    )
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    env_info = env.reset(train_mode=True)[brain_name]
    # number of agents in the environment
    print("Number of agents:", len(env_info.agents))

    # number of actions
    action_size = brain.vector_action_space_size
    print("Number of actions:", action_size)

    # examine the state space
    states = env_info.vector_observations
    state_size = states.shape[1]
    print("States have shape:", states.shape)

    agent_config = {
        "state_size": state_size,
        "action_size": action_size,
        "num_agents": len(env_info.agents),
        "random_seed": 0,
    }
    scores, avg_score_list = train_ddpg(
        env=env,
        brain=brain_name,
        agent_config=agent_config,
    )

    plt.plot(np.arange(len(scores)), scores)
    plt.plot(np.arange(len(avg_score_list)), avg_score_list)
    plt.ylabel("Score")
    plt.xlabel("Episode")
    plt.savefig("score_per_episode.png")
    plt.show()
