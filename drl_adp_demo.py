from ais.nano_rts_ai import AI, RushAI
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

lr = 2.5e-4
gamma = 0.99
gae_lambda = 0.95
clip_range = 0.2
max_clip_range = 4
ent_coef = 0.01
vf_coef = 0.5
max_grad_norm = 0.5
num_envs = 16
num_steps = 512
anneal_time = 3000
cuda = True
device = 'cuda'
adp_weight = 40
pae_length = 256
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
        
        self.value = nn.Sequential(
                layer_init(nn.Conv2d(27, 16, kernel_size=(3, 3), stride=(2, 2))),
                nn.ReLU(),
                layer_init(nn.Conv2d(16, 32, kernel_size=(2, 2))),
                nn.ReLU(),
                nn.Flatten(),
                layer_init(nn.Linear(cnn_output_size, 256)),
                nn.ReLU(), 
                layer_init(nn.Linear(256, 1), std=1)
            )
        
    def get_action_distris(self,states):
        states = states.permute((0, 3, 1, 2))
        action_distris = self.action(self.policy_network(states))
        """action_distris = torch.split(action_distris, action_space, dim=1)
        #softmax
        action_distris = [F.softmax(action_distri, dim=-1) for action_distri in action_distris]
        action_distris = torch.cat(action_distris, dim=1)"""
        return action_distris

    def get_value(self, states):
        states = states.permute((0, 3, 1, 2))
        value = self.value(states)
        return value
    
    def forward(self, states):
        distris = self.get_action_distris(states)
        value = self.get_value(states)
        return distris,value

class Agent:
    def __init__(self,net:ActorCritic) -> None:
        self.net = net
        self.num_envs = num_envs
        self.num_steps = num_steps
        self.pae_length = pae_length
        self.action_space = action_space
        self.out_comes = deque( maxlen= 1000)
        self.env = GameEnv([map_path for _ in range(self.num_envs)],rewards_wrights,max_steps = 5000)
        self.obs = self.env.reset()
        self.exps_list = [[] for _ in range(self.num_envs)]
        self.oppenent = RushAI(1,"Light", width, height)
        self.self_ai = RushAI(0,"Light", width, height)
    
    @torch.no_grad()
    def get_sample_actions(self,states, unit_masks):
        states = torch.Tensor(states)

        self_action_bias_mask = np.zeros((self.num_envs, sum(self.action_space)), dtype=np.int32)
        for i in range(self.num_envs):
            game:Game = self.env.games[i]
            action = self.self_ai.get_action(game)
            self_action_bias_mask[i] = action.action_to_one_hot(width, height)

        action_distris = self.net.get_action_distris(states)
        action_adp_distris = action_distris + torch.Tensor(self_action_bias_mask)*adp_weight

        distris = torch.split(action_adp_distris, self.action_space, dim=1)
        distris = [MaskedCategorical(dist) for dist in distris]
        
        unit_masks = torch.Tensor(unit_masks)
        distris[0].update_masks(unit_masks)
        
        units = distris[0].sample()
        action_components = [units]

        action_mask_list = self.env.get_action_masks(units.tolist())

        action_masks = torch.split(torch.Tensor(action_mask_list), self.action_space[1:], dim=1) 
        
        action_components +=  [dist.update_masks(action_mask).sample() for dist , action_mask in zip(distris[1:],action_masks)]
            
        actions = torch.stack(action_components)
        masks = torch.cat((unit_masks, torch.Tensor(action_mask_list)), 1)
        log_probs = torch.stack([dist.log_prob(aciton) for dist,aciton in zip(distris,actions)])
        
        return actions.T.cpu().numpy(), masks.cpu().numpy(),log_probs.T.cpu().numpy(),self_action_bias_mask,action_distris.cpu().numpy()
    
    def sample_env(self, check=False):  
        if check:
           step_record_dict = dict()
           rewards = []
           log_probs = [] 
        while len(self.exps_list[0]) < self.num_steps:
            #self.env.render()
            unit_mask = np.array(self.env.get_unit_masks(0)).reshape(self.num_envs, -1)
            vector_actions,mask,log_prob,bias_mask,action_d=self.get_sample_actions(self.obs, unit_mask)
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

            if check:
                rewards.append(np.mean([r[0] for r in rs]))
                log_probs.append(np.mean(log_prob))
            
            for i in range(self.num_envs):
                if done_n[i]:
                    done = True
                else:
                    done = False
                self.exps_list[i].append([self.obs[i],vector_actions[i],rs[i][0],mask[i],done,log_prob[i],bias_mask[i]])
                if check:
                    if done_n[i]:
                        if infos[i] == 0:
                            self.out_comes.append(1.0)
                        else:
                            self.out_comes.append(0.0)
                
            self.obs=next_obs

        train_exps = self.exps_list
        self.exps_list = [ exps[self.pae_length:self.num_steps] for exps in self.exps_list ]

        if check:
            mean_win_rates = np.mean(self.out_comes) if len(self.out_comes)>0 else 0.0
            print(mean_win_rates)

            step_record_dict['sum_rewards'] = np.sum(rewards)
            step_record_dict['mean_rewards'] = np.mean(rewards)
            step_record_dict['mean_log_probs'] = np.mean(log_probs)
            step_record_dict['mean_win_rates'] = mean_win_rates
            step_record_dict['mean_action_d'] = np.mean(action_d)
            step_record_dict['max_action_d'] = np.max(action_d)
            step_record_dict['std_action_d'] = np.std(action_d)
            return train_exps, step_record_dict
        
        return train_exps

class Calculator:
    def __init__(self,net:ActorCritic) -> None:
        self.net = net
        self.train_version = 0
        self.pae_length = pae_length
        
        if cuda and torch.cuda.is_available():
            self.device = torch.device('cuda', 0)
        else:
            self.device = torch.device('cpu')
        
        self.calculate_net = ActorCritic()
        self.calculate_net.to(self.device)
        self.share_optim = torch.optim.Adam(params=self.net.parameters(), lr=lr)
        self.states_list = None
        self.actions_list = None
        self.rewards_list = None
        self.dones_list = None
        self.old_log_probs_list = None
        self.marks_list = None
        self.bias_masks_list = None

    def begin_batch_train(self, samples_list: list):    
        s_states = [np.array([s[0] for s in samples]) for samples in samples_list]
        s_actions = [np.array([s[1] for s in samples]) for samples in samples_list]
        s_masks = [np.array([s[3] for s in samples]) for samples in samples_list]
        s_log_probs = [np.array([s[5] for s in samples]) for samples in samples_list]
        s_bias_masks = [np.array([s[6] for s in samples]) for samples in samples_list]
        
        s_rewards = [np.array([s[2] for s in samples]) for samples in samples_list]
        s_dones = [np.array([s[4] for s in samples]) for samples in samples_list]

        self.states = [torch.Tensor(states).to(self.device) for states in s_states]
        self.actions = [torch.Tensor(actions).to(self.device) for actions in s_actions]
        self.old_log_probs = [torch.Tensor(log_probs).to(self.device) for log_probs in s_log_probs]
        self.marks = [torch.Tensor(marks).to(self.device) for marks in s_masks]
        self.bias_masks = [torch.Tensor(bias_masks).to(self.device) for bias_masks in s_bias_masks]
        self.rewards = s_rewards
        self.dones = s_dones
        
        self.states_list = torch.cat([states[0:self.pae_length] for states in self.states])
        self.actions_list = torch.cat([actions[0:self.pae_length] for actions in self.actions])
        self.old_log_probs_list = torch.cat([old_log_probs[0:self.pae_length] for old_log_probs in self.old_log_probs])
        self.marks_list = torch.cat([marks[0:self.pae_length] for marks in self.marks])
        self.bias_masks_list = torch.cat([bias_masks[0:self.pae_length] for bias_masks in self.bias_masks])

    def calculate_samples_gae(self):
        np_advantages = []
        np_returns = []
        
        for states,rewards,dones in zip(self.states,self.rewards,self.dones):
            with torch.no_grad():
                values = self.calculate_net.get_value(states)
                            
            advantages,returns = calculate_gae(values.cpu().numpy().reshape(-1),rewards,dones,gamma,gae_lambda)
            np_advantages.extend(advantages[0:self.pae_length])
            np_returns.extend(returns[0:self.pae_length])
            
        np_advantages = np.array(np_advantages)
        np_returns = np.array(np_returns)
        
        return np_advantages, np_returns
        
    def end_batch_train(self):
        self.states_list = None
        self.actions_list = None
        self.rewards_list = None
        self.dones_list = None
        self.old_log_probs_list = None
        self.marks_list = None
        self.bias_masks_list = None

    def get_pg_loss(self,ratio,advantage):      
        clip_coef = clip_range
        max_clip_coef = max_clip_range
        positive = torch.where(ratio >= 1.0 + clip_coef, 0 * advantage,advantage)
        negtive = torch.where(ratio <= 1.0 - clip_coef,0 * advantage,torch.where(ratio >= max_clip_coef, 0 * advantage,advantage))
        return torch.where(advantage>=0,positive,negtive)*ratio
        
    def get_prob_entropy_value(self,states, actions, masks, bias_masks):
        distris = self.calculate_net.get_action_distris(states)
        distris = distris + bias_masks*adp_weight
        distris = torch.split(distris, action_space, dim=1)
        distris = [MaskedCategorical(dist) for dist in distris]
        values = self.calculate_net.get_value(states)
        action_masks = torch.split(masks, action_space, dim=1)
        distris = [dist.update_masks(mask,device=self.device) for dist,mask in zip(distris,action_masks)]
        log_probs = torch.stack([dist.log_prob(action) for dist,action in zip(distris,actions)])
        entropys = torch.stack([dist.entropy() for dist in distris])
        return log_probs.T, entropys.T, values

    def generate_grads(self):
        grad_norm = max_grad_norm
        
        self.calculate_net.load_state_dict(self.net.state_dict())
        np_advantages,np_returns = self.calculate_samples_gae()
        
        np_advantages = (np_advantages - np_advantages.mean()) / np_advantages.std()
                                                    
        advantage_list = torch.Tensor(np_advantages.reshape(-1,1)).to(self.device)    
        returns_list = torch.Tensor(np_returns.reshape(-1,1)).to(self.device)
        

        mini_batch_number = 1
        mini_batch_size = advantage_list.shape[0]

        for i in range(mini_batch_number):
            start_index = i*mini_batch_size
            end_index = (i+1)* mini_batch_size
            
            mini_states = self.states_list[start_index:end_index]
            mini_actions = self.actions_list[start_index:end_index]
            mini_masks = self.marks_list[start_index:end_index]
            mini_bias_masks = self.bias_masks_list[start_index:end_index]
            mini_old_log_probs = self.old_log_probs_list[start_index:end_index]
            
            self.calculate_net.load_state_dict(self.net.state_dict())
                
            mini_new_log_probs,mini_entropys,mini_new_values = self.get_prob_entropy_value(mini_states,mini_actions.T,mini_masks,mini_bias_masks)
                        
            mini_advantage = advantage_list[start_index:end_index]
            mini_returns = returns_list[start_index:end_index]
            
            ratio1 = torch.exp(mini_new_log_probs-mini_old_log_probs)
            pg_loss = self.get_pg_loss(ratio1,mini_advantage)

            # Policy loss
            pg_loss = -torch.mean(pg_loss)
            
            entropy_loss = -torch.mean(mini_entropys)
            
            v_loss = F.mse_loss(mini_new_values, mini_returns)

            loss = pg_loss + ent_coef * entropy_loss + v_loss*vf_coef

            self.calculate_net.zero_grad()

            loss.backward()
            
            grads = [
                param.grad.data.cpu().numpy()
                if param.grad is not None else None
                for param in self.calculate_net.parameters()
            ]
                
            # Updating network parameters
            for param, grad in zip(self.net.parameters(), grads):
                param.grad = torch.FloatTensor(grad)
                
            if grad_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(self.net.parameters(),grad_norm)
            self.share_optim.step()
    
if __name__ == "__main__":
    comment = "drl_adp_" + str(adp_weight)+"_anneal_"+str(anneal_time)
    writer = SummaryWriter(comment=comment)
    net = ActorCritic()
    agent = Agent(net)
    calculator = Calculator(net)
    MAX_VERSION = 5000
    REPEAT_TIMES = 10
    for version in range(MAX_VERSION):
        samples_list, infos = agent.sample_env(check=True)
        for (key,value) in infos.items():
                writer.add_scalar(key,value,version)

        print("version:",version,"reward:",infos["mean_rewards"],"adp_weight:",adp_weight)

        calculator.begin_batch_train(samples_list)
        for _ in range(REPEAT_TIMES):
            calculator.generate_grads()
        calculator.end_batch_train()

        #adp_weight = max(0.0, adp_weight - 1.0/anneal_time)
