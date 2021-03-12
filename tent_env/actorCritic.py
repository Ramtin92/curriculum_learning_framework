import argparse
import gym
import numpy as np
from collections import namedtuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import os

SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])

class ActorCriticPolicy(nn.Module):
    """
    implements both actor and critic in one model
    """
    def __init__(self, num_actions, input_size, hidden_layer_size, learning_rate, gamma, decay_rate,
                 epsilon, random_seed):
        super(ActorCriticPolicy, self).__init__()
        # store hyper-params
        self._A = num_actions
        self._D = input_size
        self._H = hidden_layer_size
        self.gamma = gamma
        self.decay_rate = decay_rate #?
        self.learning_rate = learning_rate
        self.random_seed = random_seed #?
        self.eps = epsilon

        self.affine1 = nn.Linear(self._D, self._H)

        # actor's layer
        self.action_head = nn.Linear(self._H, self._A)

        # critic's layer
        self.value_head = nn.Linear(self._H, 1)

        # action & reward buffer
        self.saved_actions = []
        self.rewards = []

    def forward(self, x):
        """
        forward of both actor and critic
        """
        x = F.relu(self.affine1(x))

        # actor: chooses action to take from state s_t
        action_prob = F.softmax(self.action_head(x), dim=-1)

        # critic: evaluates being in the state s_t
        state_values = self.value_head(x)

        # return values for both actor and critic as a tuple of 2 values:
        # 1. a list with the probability of each action over the action space
        # 2. the value from state s_t
        return action_prob, state_values


    def select_action(self, state):
        state = torch.from_numpy(state).float()
        probs, state_value = self(state)

        # create a categorical distribution over the list of probabilities of actions
        m = Categorical(probs)

        # and sample an action using the distribution
        action = m.sample()

        # save to action buffer
        self.saved_actions.append(SavedAction(m.log_prob(action), state_value))

        # the action to take (left or right)
        return action.item()

    def set_rewards(self, reward):
        self.rewards.append(reward)

    def finish_episode(self):
        """
        Training code. Calculates actor and critic loss and performs back propagation.
        """
        R = 0
        saved_actions = self.saved_actions
        policy_losses = [] # save actor (policy) loss
        value_losses = [] # save critic (value) loss
        returns = [] # save the true values

        # calculate the true value using rewards returned from the environment
        for r in self.rewards[::-1]:
            # calculate the discounted value
            R = r + self.gamma * R
            returns.insert(0, R)

        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + self.eps) #? eps

        for (log_prob, value), R in zip(saved_actions, returns):
            advantage = R - value.item()

            # calculate actor (policy) loss
            policy_losses.append(-log_prob * advantage)

            # calculate critic (value) loss using L1 smooth loss
            value_losses.append(F.smooth_l1_loss(value, torch.tensor([R])))

        self.optimizer = torch.optim.Adam(self.parameters(), self.learning_rate, weight_decay=1e-4) #self.net

        # reset gradients
        self.optimizer.zero_grad()

        # sum up all the values of policy_losses and value_losses
        loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()

        # perform back propagation
        loss.backward()
        self.optimizer.step()

        # reset rewards and action buffer
        del self.rewards[:]
        del self.saved_actions[:]

    def load_model(self, curriculum_no, beam_no, env_no):#
        experiment_file_name = '_c' + str(curriculum_no) + '_b' + str(beam_no) + '_e' + str(env_no)
        """
        # path_to_load = self.log_dir + os.sep + self.env_id + experiment_file_name + '.npz'
        # data = np.load(path_to_load)
        # self._model['W1'] = data['layer1']
        # self._model['W2'] = data['layer2'] 
        """
        self.load_state_dict(torch.load(experiment_file_name))


    def save_model(self, curriculum_no, beam_no, env_no):#
        experiment_file_name = '_c' + str(curriculum_no) + '_b' + str(beam_no) + '_e' + str(env_no)
        """
        # path_to_save = self.log_dir + os.sep + self.env_id + experiment_file_name + '.npz'
        # np.savez(path_to_save, layer1=self._model['W1'], layer2=self._model['W2'])
        # print("saved to: ", path_to_save)
        """
        torch.save(self.state_dict(), experiment_file_name)



def main():

    parser = argparse.ArgumentParser(description='PyTorch actor-critic example')
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                        help='discount factor (default: 0.99)')
    parser.add_argument('--seed', type=int, default=543, metavar='N',
                        help='random seed (default: 543)')
    parser.add_argument('--render', action='store_false',
                        help='render the environment')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='interval between training status logs (default: 10)')
    args = parser.parse_args()


if __name__ == '__main__':
    main()

