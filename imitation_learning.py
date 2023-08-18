from ais.nano_rts_ai import RushAI, RandomAI
from nanorts.game import Game
from nanorts.game_env import GameEnv

import torch 
import torch.nn as nn
from collections import deque
import numpy as np
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import argparse

from utils import layer_init, calculate_gae, MaskedCategorical

lr = 1e-4
num_envs = 16
num_steps = 1024
cuda = True
device = 'cuda'
grad_norm_max = 0.5
mini_batch_size = 1024

rewards_wrights = {'win': 10,'harvest': 1,'return': 1,'attack': 1, 'produce_worker': 1, 
                   'produce_light': 4, 'produce_heavy': 4, 'produce_ranged': 4, 'produce_base': 0, 'produce_barracks': 0.2}

map_path = 'maps\\8x8\\bases8x8.xml'
map = '16x16'
if map == '8x8':
    width = 8
    height = 8
    cnn_output_size = 32*2*2
    map_path = 'maps\\8x8\\bases8x8.xml'
if map == '16x16':
    width = 16
    height = 16
    cnn_output_size = 32*6*6
    map_path = 'maps\\16x16\\basesWorkers16x16.xml'
action_space = [width*height, 6, 4, 4, 4, 4, 7, 49]
observation_space = [height,width,27]

#main network
class ActorCritic(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.policy_network = nn.Sequential(
            layer_init(nn.Conv2d(27, 16, kernel_size=(3, 3), stride=(2, 2))),
            nn.ReLU(),
            layer_init(nn.Conv2d(16, 32, kernel_size=(2, 2))),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(cnn_output_size, 256)),
            nn.ReLU(),
        )

        self.action = layer_init(nn.Linear(256, sum(action_space)))
        
    def get_action_distris(self,states:torch.Tensor):
        states = states.permute((0, 3, 1, 2))
        # float states
        states = states.float()
        action_distris = self.action(self.policy_network(states))
        """action_distris = torch.split(action_distris, action_space, dim=1)
        #softmax
        action_distris = [F.softmax(action_distri, dim=-1) for action_distri in action_distris]
        action_distris = torch.cat(action_distris, dim=1)"""
        return action_distris

    def forward(self, states):
        distris = self.get_action_distris(states)
        return distris

class Agent:
    def __init__(self,net:ActorCritic) -> None:
        self.net = net
        self.num_envs = num_envs
        self.num_steps = num_steps
        self.action_space = action_space
        self.out_comes = deque( maxlen= 1000)
        self.env = GameEnv([map_path for _ in range(self.num_envs)],rewards_wrights,max_steps = 5000)
        self.obs = self.env.reset()
        self.oppenent = RandomAI(1)
        self.self_ai = RushAI(0,"Light", width, height)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
    
    @torch.no_grad()
    def get_sample_actions(self,states, unit_masks):
        states = torch.Tensor(states)

        imitation_actions = np.zeros((self.num_envs, sum(self.action_space)), dtype=np.int32)
        for i in range(self.num_envs):
            game:Game = self.env.games[i]
            action = self.self_ai.get_action(game)
            imitation_actions[i] = action.action_to_one_hot2(width, height)

        action_distris = self.net.get_action_distris(states)

        distris = torch.split(action_distris, self.action_space, dim=1)
        distris = [MaskedCategorical(dist) for dist in distris]
        
        unit_masks = torch.Tensor(unit_masks)
        distris[0].update_masks(unit_masks)
        
        units = distris[0].sample()
        action_components = [units]

        action_mask_list = self.env.get_action_masks(units.tolist())

        action_masks = torch.split(torch.Tensor(action_mask_list), self.action_space[1:], dim=1) 
        
        action_components +=  [dist.update_masks(action_mask).sample() for dist , action_mask in zip(distris[1:],action_masks)]
            
        actions = torch.stack(action_components)
        log_probs = torch.stack([dist.log_prob(aciton) for dist,aciton in zip(distris,actions)])
        
        return actions.T.cpu().numpy(), log_probs.T.cpu().numpy(),imitation_actions,action_distris
    
    def sample(self):  
        step_record_dict = dict()
        rewards = []
        log_probs = []

        states = []
        teacher_actions = []
        while len(teacher_actions) < self.num_steps:
            #self.env.render()
            unit_mask = np.array(self.env.get_unit_masks(0)).reshape(self.num_envs, -1)
            vector_actions,log_prob,bias_mask,action_d=self.get_sample_actions(self.obs, unit_mask)
            teacher_actions.append(bias_mask)
            states.append(self.obs)
            
            actions0 = []
            actions1 = []
            for i in range(self.num_envs):
                game:Game = self.env.games[i]
                vector_action = vector_actions[i]
                oppe_action = self.oppenent.get_action(self.env.games[i])
                action = game.vector_to_action(vector_action)
                actions0.append(action)
                actions1.append(oppe_action)
            next_obs, rs, done_n, infos = self.env.step(actions0, actions1)

            rewards.append(np.mean([r[0] for r in rs]))
            log_probs.append(np.mean(log_prob))
            
            for i in range(self.num_envs):
                if done_n[i]:
                    if infos[i] == 0:
                        self.out_comes.append(1.0)
                    else:
                        self.out_comes.append(0.0)
                
            self.obs=next_obs

        teacher_actions = torch.tensor(teacher_actions).reshape(-1, sum(self.action_space))
        states = torch.tensor(states).reshape(-1, height, width, 27)

        
        mean_win_rates = np.mean(self.out_comes) if len(self.out_comes)>0 else 0.0
        print(mean_win_rates)

        step_record_dict['sum_rewards'] = np.sum(rewards)
        step_record_dict['mean_rewards'] = np.mean(rewards)
        step_record_dict['mean_log_probs'] = np.mean(log_probs)
        step_record_dict['mean_win_rates'] = mean_win_rates
        return step_record_dict, states, teacher_actions

class Calculator:
    def __init__(self, net:ActorCritic):
        self.net = net
        self.calculate_net = ActorCritic()
        self.calculate_net.to(device)
        self.share_optim = torch.optim.Adam(params=self.net.parameters(), lr=lr)

    def train(self, states, teacher_actions):
        self.calculate_net.load_state_dict(self.net.state_dict())

        net_action_dist = self.calculate_net(states.to(device))
        """net_action_dist = torch.split(net_action_dist, action_space, dim=1)
        #softmax
        net_action_dist = [F.softmax(action_distri, dim=-1) for action_distri in net_action_dist]
        net_action_dist = torch.cat(net_action_dist, dim=1)"""

        teacher_actions = teacher_actions.to(device).float()
        
        """n_mini_batches = int(np.ceil(states.shape[0] / mini_batch_size))

        for index in range(n_mini_batches):
            start = index * mini_batch_size
            end = start + mini_batch_size
            mini_batch_teacher_actions = teacher_actions[start:end].to(device)
            mini_batch_net_action_dist = net_action_dist[start:end].to(device)
            loss = F.mse_loss(mini_batch_net_action_dist, mini_batch_teacher_actions)
            self.share_optim.zero_grad()
            loss.backward(retain_graph=True)

            grads = [
                    param.grad.data.cpu().numpy()
                    if param.grad is not None else None
                    for param in self.calculate_net.parameters()
                ]
                    
            # Updating network parameters
            for param, grad in zip(self.net.parameters(), grads):
                param.grad = torch.FloatTensor(grad)
            
            nn.utils.clip_grad_norm_(self.net.parameters(), grad_norm_max)

            self.share_optim.step()"""
        
        loss = F.mse_loss(net_action_dist, teacher_actions)
        self.share_optim.zero_grad()
        loss.backward()

        grads = [
                param.grad.data.cpu().numpy()
                if param.grad is not None else None
                for param in self.calculate_net.parameters()
            ]
                
        # Updating network parameters
        for param, grad in zip(self.net.parameters(), grads):
            param.grad = torch.FloatTensor(grad)
        
        nn.utils.clip_grad_norm_(self.net.parameters(), grad_norm_max)

        self.share_optim.step()

        
if __name__ == "__main__":
    comment = "imitation_learning"
    writer = SummaryWriter(comment=comment)
    net = ActorCritic()
    agent = Agent(net)
    calculator = Calculator(net)
    MAX_VERSION = 5000
    REPEAT_TIMES = 4
    for version in range(MAX_VERSION):
        step_record_dict, states, teacher_actions = agent.sample()
        for key, value in step_record_dict.items():
            writer.add_scalar(key, value, version)
        print("version:",version,"reward:", step_record_dict["mean_rewards"])
        for _ in range(REPEAT_TIMES):
            calculator.train(states, teacher_actions)
        