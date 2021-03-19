import numpy as np
from enviroments import get_envs
from agents import ActorCriticPolicy
from args import Args
import random
import time


def check_training_done_callback(reward_array, done_array):
    done_cond = 0
    reward_cond = 0
    if len(done_array) > 30:
        if np.mean(done_array[-10:]) > 0.85 and np.mean(done_array[-40:]) > 0.85:
            if abs(np.mean(done_array[-40:]) - np.mean(done_array[-10:])) < 0.5:
                done_cond = 1

        if done_cond:
            if np.mean(reward_array[-40:]) > 950:
                reward_cond = 1

        if done_cond and reward_cond:
            return 1
        else:
            return 0
    else:
        return 0


def train(args, env, agent, index_env, is_final_env):  # fill in more args if it's needed
    env.reset()
    episode = 0
    time_step = 0
    reward_sum = 0
    done_arr = []
    curr_task_completion_array = []
    reward_arr = []
    avg_reward = []
    while True:

        env.render()
        time.sleep(.1)
        obs = env.get_observation()
        a = agent.select_action(obs)

        new_obs, reward, done, info = env.step(a)

        agent.set_rewards(reward)
        reward_sum += reward

        time_step += 1

        if time_step > args.time_limit or done:

            # finish agent
            if done:
                done_arr.append(1)
                curr_task_completion_array.append(1)
            elif time_step > args.time_limit:
                done_arr.append(0)
                curr_task_completion_array.append(0)

            print("\n\nfinished episode = " + str(episode) + " with " + str(reward_sum) + "\n")

            reward_arr.append(reward_sum)
            avg_reward.append(np.mean(reward_arr[-40:]))

            done = 1
            agent.finish_episode()

            episode += 1
            time_step = 0

            env.reset()
            reward_sum = 0

            env_flag = 0

            if is_final_env:
                env_flag = check_training_done_callback(reward_arr, done_arr)

            # quit after some number of episodes
            if episode > 120000 or env_flag == 1:
                agent.save_model(0, 0, index_env)
                # total_episodes_arr.append(episode)
                break

    return reward_arr, avg_reward, time_step, index_env


def main(args):
    random.seed(args.seed)
    envs = get_envs()
    results = []
    agent = ActorCriticPolicy(args.num_actions,
                              args.input_size,
                              args.hidden_size,
                              args.learning_rate,
                              args.gamma,
                              args.decay_rate,
                              args.epsilon)

    is_final_env = 0
    for index_env, env in enumerate(envs):
        agent.reset()
        if index_env > 0:
            agent.load_model(0, 0, index_env-1)
            agent.reinit()

        if index_env == len(envs) - 1:
            is_final_env = 1

        result = train(args, env, agent, index_env, is_final_env)
        results.append(result)

    # TODO write and save


if __name__ == '__main__':
    opt = Args()
    opt = opt.update_args()
    main(opt)
