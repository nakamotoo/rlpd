from typing import Dict

import gym
import numpy as np

from rlpd.wrappers.wandb_video import WANDBVideo


def evaluate(
    agent, env: gym.Env, num_episodes: int, save_video: bool = False
) -> Dict[str, float]:
    if save_video:
        env = WANDBVideo(env, name="eval_video", max_videos=1)
    env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=num_episodes)
    goal_achieved_list=[]
    for i in range(num_episodes):
        observation, done = env.reset(), False
        goal_achieved=0
        while not done:
            action = agent.eval_actions(observation)
            observation, _, done, info = env.step(action)
            if info["goal_achieved"]:
                goal_achieved=1
        goal_achieved_list.append(goal_achieved)
    return {"return": np.mean(env.return_queue), "length": np.mean(env.length_queue), "goal_achieved_rate": np.mean(goal_achieved_list)}
