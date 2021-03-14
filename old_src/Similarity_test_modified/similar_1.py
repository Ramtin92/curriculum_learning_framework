import gym
import numpy as np
import time
import os
import sys
import gym_novel_gridworlds
from SimpleDQN import SimpleDQN
import matplotlib.pyplot as plt

from actorCritic import ActorCriticPolicy


def CheckTrainingDoneCallback(reward_array, done_array, env):
    done_cond = False
    reward_cond = False
    if len(done_array) > 30:
        if np.mean(done_array[-10:]) > 0.85 and np.mean(done_array[-40:]) > 0.85:
            if abs(np.mean(done_array[-40:]) - np.mean(done_array[-10:])) < 0.5:
                done_cond = True

        if done_cond == True:
            if env < 3:
                if np.mean(reward_array[-10:]) > 950:
                    reward_cond = True
            else:
                if np.mean(reward_array[-10:]) > 950:
                    reward_cond = True

        if done_cond == True and reward_cond == True:
            return 1
        else:
            return 0
    else:
        return 0


if __name__ == "__main__":

    no_of_environmets = 1

    width_array = [10]
    height_array = [10]
    no_trees_array = [1]
    no_rocks_array = [0]
    crafting_table_array = [1]
    starting_trees_array = [0]
    starting_rocks_array = [1]
    type_of_env_array = [1]

    total_timesteps_array = []
    total_reward_array = []
    avg_reward_array = []
    final_timesteps_array = []
    final_reward_array = []
    final_avg_reward_array = []
    curr_task_completion_array = []
    final_task_completion_array = []

    actionCnt = 5
    D = 37  # 8 beams x 4 items Lidar + 5 inventory items
    NUM_HIDDEN = 10
    DECAY_RATE = 0.99
    MAX_EPSILON = 0.1
    GAMMA = 0.95
    LEARNING_RATE = 1e-3
    random_seed = 42
    total_num_eps = 1000
    EPISODES = 100

    # agent = SimpleDQN(actionCnt,D,NUM_HIDDEN,LEARNING_RATE,GAMMA,DECAY_RATE,MAX_EPSILON,random_seed)
    # agent.set_explore_epsilon(MAX_EPSILON)

    action_space = ['W', 'A', 'D', 'U', 'C']
    total_episodes_arr = []

    for i in range(no_of_environmets):

        width = width_array[i]
        height = height_array[i]
        no_trees = no_trees_array[i]
        no_rocks = no_rocks_array[i]
        crafting_table = crafting_table_array[i]
        starting_trees = starting_trees_array[i]
        starting_rocks = starting_rocks_array[i]
        type_of_env = type_of_env_array[i]

        final_status = False

        if i == 0:
            # agent = SimpleDQN(actionCnt, D, NUM_HIDDEN, LEARNING_RATE, GAMMA, DECAY_RATE, MAX_EPSILON, random_seed)
            # agent.set_explore_epsilon(MAX_EPSILON)

            agent = ActorCriticPolicy(D, NUM_HIDDEN, GAMMA, LEARNING_RATE, random_seed)

        else:
            # agent = SimpleDQN(actionCnt, D, NUM_HIDDEN, LEARNING_RATE, GAMMA, DECAY_RATE, MAX_EPSILON, random_seed)
            # agent.set_explore_epsilon(MAX_EPSILON)

            agent = ActorCriticPolicy(D, NUM_HIDDEN, GAMMA, LEARNING_RATE, random_seed)

            agent.load_model(0, 0, i - 1)
            agent.reset()
            print("loaded model")

        if i == no_of_environmets - 1:
            final_status = True

        env_id = 'NovelGridworld-v0'
        env = gym.make(env_id, map_width=width, map_height=height,
                       items_quantity={'tree': no_trees, 'rock': no_rocks, 'crafting_table': crafting_table,
                                       'pogo_stick': 0},
                       initial_inventory={'wall': 0, 'tree': starting_trees, 'rock': starting_rocks,
                                          'crafting_table': 0, 'pogo_stick': 0}, goal_env=type_of_env,
                       is_final=final_status)

        for i_episode in range(EPISODES):

            # reset environment and episode reward
            state = env.reset()
            ep_reward = 0

            # for each episode, only run 9999 steps so that we don't infinite loop while learning
            for t in range(1, 10000):

                # select action from policy
                action = agent.select_action(state)

                # take the action
                state, reward, done, _ = env.step(action)


                env.render()

                agent.rewards.append(reward)
                ep_reward += reward
                if done:
                    break

            # update cumulative reward
            running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward

            # perform back propagation
            agent.finish_episode()

            # log results
            if i_episode % 2 == 0:
                print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                    i_episode, ep_reward, running_reward))

            # check if we have "solved" the problem
            if running_reward > env.spec.reward_threshold:
                print("Solved! Running reward is now {} and "
                      "the last episode runs to {} time steps!".format(running_reward, t))
                break

    #     t_step = 0
    #     episode = 0
    #     t_limit = 100
    #     reward_sum = 0
    #     reward_arr = []
    #     avg_reward = []
    #     done_arr = []
    #     env_flag = 0
    #
    #     env.reset()
    #
    #     while True:
    #
    #         # get obseration from sensor
    #         obs = env.get_observation()
    #
    #         # act
    #         # a = agent.process_step(obs, True)
    #         a = agent.select_action(obs)
    #
    #
    #         new_obs, reward, done, info = env.step(a)
    #
    #         # give reward
    #         agent.give_reward(reward)
    #         reward_sum += reward
    #
    #         t_step += 1
    #
    #         if t_step > t_limit or done == True:
    #
    #             # finish agent
    #             if done == True:
    #                 done_arr.append(1)
    #                 curr_task_completion_array.append(1)
    #             elif t_step > t_limit:
    #                 done_arr.append(0)
    #                 curr_task_completion_array.append(0)
    #
    #             print("\n\nfinished episode = " + str(episode) + " with " + str(reward_sum) + "\n")
    #
    #             reward_arr.append(reward_sum)
    #             avg_reward.append(np.mean(reward_arr[-40:]))
    #
    #             total_reward_array.append(reward_sum)
    #             avg_reward_array.append(np.mean(reward_arr[-40:]))
    #             total_timesteps_array.append(t_step)
    #
    #             done = True
    #             t_step = 0
    #             agent.finish_episode()
    #
    #             # update after every episode
    #             if episode % 10 == 0:
    #                 agent.update_parameters()
    #
    #             # reset environment
    #             episode += 1
    #
    #             env.reset()
    #             reward_sum = 0
    #
    #             env_flag = 0
    #             # if i< 3:
    #             # 	env_flag = CheckTrainingDoneCallback(reward_arr, done_arr, i)
    #
    #             # quit after some number of episodes
    #             if episode > total_num_eps or env_flag == 1:
    #                 agent.save_model(0, 0, i)
    #                 total_episodes_arr.append(episode)
    #
    #                 break
    #
    # np.save("sim1.npz", agent.grads)

