import numpy as np
from enviroments import get_envs
from agents import ActorCriticPolicy
from args import Args



def CheckTrainingDoneCallback(reward_array, done_array, env):
    done_cond = False
    reward_cond = False
    if len(done_array) > 30:
        if np.mean(done_array[-10:]) > 0.85 and np.mean(done_array[-40:]) > 0.85:
            if abs(np.mean(done_array[-40:]) - np.mean(done_array[-10:])) < 0.5:
                done_cond = True

        if done_cond == True:
            if env < 3:
                if np.mean(reward_array[-40:]) > 950:
                    reward_cond = True

        if done_cond == True and reward_cond == True:
            return 1
        else:
            return 0
    else:
        return 0

def train(args, env, agent): # fill in more args if it's needed
    env.reset()
    episode = 0
    time_step = 0
    reward_sum = 0
    done_arr = []
    curr_task_completion_array = []
    reward_arr = []
    avg_reward = []
    while True:
        obs = env.get_observation()
        a = agent.select_action(obs)

        print("obs", obs)
        print("a", a)

        new_obs, reward, done, info = env.step(a)

        print("new_obs", new_obs)
        print("reward", reward)
        print("done", done)

        agent.set_rewards(reward)
        reward_sum += reward
        print("reward_sum", reward_sum)

        time_step += 1

        if time_step > args.time_limit or done == True:

            # finish agent

            done_arr.append(done)
            curr_task_completion_array.append(done)


            print("\n\nfinished episode = " + str(episode) + " with " + str(reward_sum) + "\n")

            reward_arr.append(reward_sum)
            avg_reward.append(np.mean(reward_arr[-40:]))

            # total_reward_array.append(reward_sum)
            # avg_reward_array.append(np.mean(reward_arr[-40:]))
            # total_timesteps_array.append(time_step)

            t_step = 0
            agent.finish_episode()

            # reset environment
            episode += 1

            env.reset()
            reward_sum = 0

            env_flag = 0

            if env.is_final:
                env_flag = CheckTrainingDoneCallback(reward_arr, done_arr, i)

            # quit after some number of episodes
            if episode > 120000 or env_flag == 1:
                agent.save_model(0, 0, i)
                # total_episodes_arr.append(episode)

                break

    return None #TODO: return training history to the main function, better be a dictionary

def main(args):
    envs = get_envs()
    results = []
    agent = ActorCriticPolicy(args.num_actions,
                              args.input_size,
                              args.hidden_size,
                              args.learning_rate,
                              args.gamma,
                              args.decay_rate,
                              args.max_epsilon,
                              args.seed)
    for env in envs:
        agent.reset()
        result = train(args, env, agent)
        results.append(result)

    # TODO write and save

if __name__ == '__main__':
    opt = Args()
    opt = opt.update_args()
    main(opt)
