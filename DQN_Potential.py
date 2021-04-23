import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
import matplotlib.pyplot as plt
import copy
from enviroments import get_envs
import os
import time

# hyper-parameters
BATCH_SIZE = 128
LR = 0.001
GAMMA = 0.99
EPISILO = 0.9
MEMORY_CAPACITY = 2000
Q_NETWORK_ITERATION = 100


envs = get_envs()
env = envs[2]

NUM_ACTIONS = env.action_space.n
NUM_STATES = env.observation_space.shape[0]
ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(), int) else env.action_space.sample.shape

class Net(nn.Module):
    """docstring for Net"""
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(NUM_STATES, 16)
        self.fc1.weight.data.normal_(0,0.1)
        self.out = nn.Linear(16,NUM_ACTIONS)
        self.out.weight.data.normal_(0,0.1)

    def forward(self,x):
        x = self.fc1(x)
        x = F.relu(x)
        action_prob = self.out(x)
        return action_prob

class DQN():
    """docstring for DQN"""
    def __init__(self, potential_based):
        super(DQN, self).__init__()
        self.eval_net, self.target_net = Net(), Net()

        self.learn_step_counter = 0
        self.memory_counter = 0
        self.memory = np.zeros((MEMORY_CAPACITY, NUM_STATES * 2 + 2))
        # why the NUM_STATE*2 +2
        # When we store the memory, we put the state, action, reward and next_state in the memory
        # here reward and action is a number, state is a ndarray
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()
        self.log_dir = 'results'
        os.makedirs(self.log_dir, exist_ok = True)
        if potential_based == 1:
            self.prev_eval_net, self.prev_target_net = Net(), Net()
            self.load_model()

    def choose_action(self, state):
        state = torch.unsqueeze(torch.FloatTensor(state), 0) # get a 1D array
        if np.random.randn() <= EPISILO:# greedy policy
            action_value = self.eval_net.forward(state)
            action = torch.max(action_value, 1)[1].data.numpy()
            # tp = action_value[0][2]
            action = action[0] if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)
        else: # random policy
            action = np.random.randint(0,NUM_ACTIONS)
            action = action if ENV_A_SHAPE ==0 else action.reshape(ENV_A_SHAPE)
        return action

    def return_critic(self,state):
        state = torch.unsqueeze(torch.FloatTensor(state), 0) # get a 1D array
        action_value = self.prev_eval_net.forward(state)
        action = torch.max(action_value, 1)[1].data.numpy()
        action = action[0] if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)
        critic_value = action_value[0][action]
        return critic_value.data.numpy()

    def store_transition(self, state, action, reward, next_state):
        transition = np.hstack((state, [action, reward], next_state))
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1


    def learn(self):

        #update the parameters
        if self.learn_step_counter % Q_NETWORK_ITERATION ==0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter+=1

        #sample batch from memory
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        batch_memory = self.memory[sample_index, :]
        batch_state = torch.FloatTensor(batch_memory[:, :NUM_STATES])
        batch_action = torch.LongTensor(batch_memory[:, NUM_STATES:NUM_STATES+1].astype(int))
        batch_reward = torch.FloatTensor(batch_memory[:, NUM_STATES+1:NUM_STATES+2])
        batch_next_state = torch.FloatTensor(batch_memory[:,-NUM_STATES:])

        #q_eval
        q_eval = self.eval_net(batch_state).gather(1, batch_action)
        q_next = self.target_net(batch_next_state).detach()
        q_target = batch_reward + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save_model(self):        
        path_to_save_eval = self.log_dir + os.sep +  'model_1_eval' + '.pt'
        path_to_save_target = self.log_dir + os.sep +  'model_1_target' + '.pt'
        torch.save(self.eval_net, path_to_save_eval)
        torch.save(self.target_net, path_to_save_target)
    
    def load_model(self):
        path_to_load_eval = self.log_dir + os.sep +  'model_1_eval' + '.pt'
        path_to_load_target = self.log_dir + os.sep +  'model_1_target' + '.pt'
        self.prev_eval_net = torch.load(path_to_load_eval)
        self.prev_target_net = torch.load(path_to_load_target)
        print("loaded model")


def main():
    potential_based_env = 1 # 0 when it is source and 1 when it is target
    dqn = DQN(potential_based_env)
    episodes = 30000
    print("Collecting Experience....")
    reward_list = []
    episode_arr = []
    done_list = []
    done_mean_arr = []
    reward_mean_arr = []
    plt.ion()
    fig, ax = plt.subplots()
    for i in range(episodes):
        state = env.reset()
        ep_reward = 0
        timestep = 0
        while True:
            action = dqn.choose_action(state)
            next_state, reward , done, info = env.step(action)
            timestep += 1

            if potential_based_env == 1:
                if timestep > 0 and timestep < 100 and done == False:
                    potential_next_state = dqn.return_critic(next_state)
                    potential_current_state = dqn.return_critic(state)
                    potential_reward = GAMMA*potential_next_state - potential_current_state
                    reward += potential_reward/100 #Checking how potential reward is shaping the actual reward

            dqn.store_transition(state, action, reward, next_state)
            ep_reward += reward

            if dqn.memory_counter >= MEMORY_CAPACITY:
                dqn.learn()
                if done or timestep == 100:
                    print("episode: {} , the episode reward is {}, and done is {}".format(i, round(ep_reward, 3), done))
            if done or timestep == 100:
                break
            state = next_state
        r = copy.copy(ep_reward)
        reward_list.append(r)
        episode_arr.append(i)
        if done == True:
            done_list.append(1)
        else:
            done_list.append(0)
        reward_mean_arr.append(np.mean(reward_list[-30:]))
        done_mean_arr.append(np.mean(done_list[-30:]))


    dqn.save_model()
    plt.xlabel('Episodes')
    plt.ylabel('Task completion rate')
    plt.plot(episode_arr, done_mean_arr)
    path_to_save_done_fig = "DQN_done_flag_" + str(potential_based_env) + "_.png"
    plt.savefig(path_to_save_done_fig)
    plt.clf()
    plt.xlabel('Episodes')
    plt.ylabel('Average Reward rate')    
    plt.plot(episode_arr, reward_mean_arr)
    path_to_save_reward_fig = "DQN_reward_flag_" + str(potential_based_env) + "_.png"
    plt.savefig(path_to_save_reward_fig)

if __name__ == '__main__':
    main()
