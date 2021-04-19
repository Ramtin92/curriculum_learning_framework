import numpy as np
from enviroments import get_envs
import os
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
            reward_cond = 1
            # if np.mean(reward_array[-40:]) > 910:

        if done_cond and reward_cond:
            return 1
        else:
            return 0
    else:
        return 0


def train(args, env, agent,index_env, is_final_env, agent_prev = None):  # fill in more args if it's needed
    env.reset()
    episode = 0
    time_step = 0
    reward_sum = 0
    done_arr = []
    curr_task_completion_array = []
    reward_arr = []
    avg_reward = []
    timestep_arr = []
    episode_arr = []
    while True:
        # env.render()
        # time.sleep(.1)

        obs = env.get_observation()
        # prev_obs = obs
        a = agent.select_action(obs)

        new_obs, reward, done, info = env.step(a)

        if index_env > 0 and time_step > 0:
            potential_next_state = agent_prev.return_critic_value(new_obs)
            potential_current_state = agent_prev.return_critic_value(obs)
            potential_reward = args.gamma*potential_next_state - potential_current_state
            agent.rewards[-1] += potential_reward[0]*100 #Factor to make potential_reward in same decimals if reward = -1 potential_reward should be comparable
            # print(agent.rewards)
            # if time_step == 0:
            #     reward_sum += reward
            if time_step > 0 and done == False:
                reward_sum += agent.rewards[-1]
            else: # when done == true, the reward is 1000, and this is the last time step
                reward_sum += reward
        else:
            reward_sum += reward

        agent.set_rewards(reward)
        time_step += 1  

        if time_step > args.time_limit or done:

            # finish agent
            if done:
                done_arr.append(1)
                curr_task_completion_array.append(1)
            elif time_step > args.time_limit:
                done_arr.append(0)
                curr_task_completion_array.append(0)

            print("\n\nfinished episode = " + str(episode) + " with " + str(reward_sum) + " done = " + str(done) + "\n")

            reward_arr.append(reward_sum)
            avg_reward.append(np.mean(reward_arr[-40:]))
            timestep_arr.append(time_step)

            done = 1
            agent.finish_episode()

            episode += 1
            time_step = 0

            env.reset()
            reward_sum = 0

            env_flag = 0

            if not is_final_env:
                env_flag = check_training_done_callback(reward_arr, done_arr)

            # quit after some number of episodes
            if episode > args.episodes_per_task or env_flag == 1:
                agent.save_model(0, 0, index_env)
                episode_arr.append(episode)
                print("Saved")
                time.sleep(5.0)
                break

    return reward_arr, avg_reward, timestep_arr, episode_arr, index_env


def main(args):
    random.seed(args.seed)
    envs = get_envs()
    results = {'reward':[], 'avg_reward':[],'timesteps':[],'episodes_per_task':[]}
    agent = ActorCriticPolicy(args.num_actions,
                              args.input_size,
                              args.hidden_size,
                              args.learning_rate,
                              args.gamma,
                              args.decay_rate,
                              args.epsilon)

    agent_prev = ActorCriticPolicy(args.num_actions,
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
            agent_prev.load_model(0, 0, index_env-1)
            agent.eval()
            # agent.reinit()

        if index_env == len(envs) - 1:
            is_final_env = 1

        result = train(args, env, agent, index_env, is_final_env, agent_prev)
        results['reward'].extend(result[0])
        results['avg_reward'].extend(result[1])
        results['timesteps'].extend(result[2])
        results['episodes_per_task'].extend(result[3])

    log_dir = 'logs_' + str(args.seed) 
    os.makedirs(log_dir, exist_ok=True)
    path_to_save_total_reward = log_dir + os.sep + "randomseed_" + str(args.seed) + "_reward_.npz"
    np.savez_compressed(path_to_save_total_reward, curriculum_reward = np.asarray(results["reward"]))

    path_to_save_avg_reward = log_dir + os.sep + "randomseed_" + str(args.seed) + "_avg_reward_.npz"
    np.savez_compressed(path_to_save_avg_reward, curriculum_avg_reward = np.asarray(results["avg_reward"]))

    path_to_save_timesteps = log_dir + os.sep + "randomseed_" + str(args.seed) + "_timesteps_.npz"
    np.savez_compressed(path_to_save_timesteps, curriculum_reward = np.asarray(results['timesteps']))

    path_to_save_episodes = log_dir + os.sep + "randomseed_" + str(args.seed) + "_episodes_.npz"
    np.savez_compressed(path_to_save_episodes, curriculum_reward = np.asarray(results["episodes_per_task"]))


if __name__ == '__main__':
    opt = Args()
    opt = opt.update_args()
    main(opt)
