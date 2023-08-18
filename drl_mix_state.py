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
seed = 1
num_envs = 16
num_steps = 512
cuda = True
device = 'cuda'
pae_length = 256
rewards_wrights = {'win': 10,'harvest': 1,'return': 1,'attack': 1, 'produce_worker': 1, 
                   'produce_light': 4, 'produce_heavy': 4, 'produce_ranged': 4, 'produce_base': 0, 'produce_barracks': 0.2}

padding = 3
n_unit = 16

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
        self.encoder_cnn = nn.Sequential(
            layer_init(nn.Conv2d(27, 16, kernel_size=(3, 3), stride=(2, 2))),
            nn.ReLU(),
            layer_init(nn.Conv2d(16, 32, kernel_size=(2, 2))),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(32*2*2, 128)),
            nn.ReLU(),
        )

        self.encoder = nn.GRU(29, 128, batch_first=True)
        self.decoder = nn.GRU(128, 128, batch_first=True)

        self.policy_type = layer_init(nn.Linear(256, 6), std=0.01)
        self.policy_move = layer_init(nn.Linear(256, 4), std=0.01)
        self.policy_harvest = layer_init(nn.Linear(256, 4), std=0.01)
        self.policy_return = layer_init(nn.Linear(256, 4), std=0.01)
        self.policy_produce = layer_init(nn.Linear(256, 4), std=0.01)
        self.policy_produce_type = layer_init(nn.Linear(256, 7), std=0.01)
        self.policy_attack = layer_init(nn.Linear(256, 49), std=0.01)
        
        self.value = layer_init(nn.Linear(256, 1), std=1)
        
    def forward(self,cnn_states,linears_states):
        cnn_states = cnn_states.permute((0, 3, 1, 2))
        z_cnn = self.encoder_cnn(cnn_states)

        batch_size = linears_states.size(0)
        seq_len = linears_states.size(1)

        # Encoder
        encoder_outputs, hidden = self.encoder(linears_states)

        # Decoder
        decoder_state = hidden
        decoder_input = torch.zeros((batch_size, 1, 128)).to(linears_states.device)
        decoder_outputs = []

        for t in range(seq_len):
            decoder_output, decoder_state = self.decoder(decoder_input, decoder_state)
            decoder_outputs.append(decoder_output)
            decoder_input = decoder_output

        z_pn = decoder_outputs[-1][:,-1,:]

        policy_network = torch.cat((z_cnn,z_pn),dim=1)

        type_distris = MaskedCategorical(self.policy_type(policy_network))
        move_distris = MaskedCategorical(self.policy_move(policy_network))
        harvest_distris = MaskedCategorical(self.policy_harvest(policy_network))
        return_distris = MaskedCategorical(self.policy_return(policy_network))
        produce_distris = MaskedCategorical(self.policy_produce(policy_network))
        produce_type_distris = MaskedCategorical(self.policy_produce_type(policy_network))
        attack_distris = MaskedCategorical(self.policy_attack(policy_network))

        value = self.value(policy_network)

        return [type_distris,move_distris,harvest_distris,return_distris,produce_distris,produce_type_distris,attack_distris],value

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
    
    @torch.no_grad()
    def get_sample_actions(self,cnn_states, linear_states, units_pos_list):
        distris,_ = self.net(cnn_states, linear_states)
        
        action_components = [torch.Tensor(units_pos_list)]

        action_mask_list = self.env.get_action_masks(units_pos_list)

        action_masks = torch.split(torch.Tensor(action_mask_list), self.action_space[1:], dim=1) 
        
        action_components +=  [dist.update_masks(action_mask).sample() for dist , action_mask in zip(distris,action_masks)]
            
        actions = torch.stack(action_components)
        masks = torch.Tensor(action_mask_list)
        log_probs = torch.stack([dist.log_prob(aciton) for dist,aciton in zip(distris,actions[1:])])
        
        return actions.T.cpu().numpy(), masks.cpu().numpy(),log_probs.T.cpu().numpy()
    
    def sample_env(self, check=False):  
        if check:
           step_record_dict = dict()
           rewards = []
           log_probs = [] 
        while len(self.exps_list[0]) < self.num_steps:
            self.env.render()
            unit_mask = np.array(self.env.get_unit_masks(0)).reshape(self.num_envs, -1)

            units_pos_list = []
            for unit_mask in unit_mask:
                if np.sum(unit_mask) == 0:
                    units_pos_list.append(-1)
                else:
                    units_pos_list.append(np.random.choice(np.where(unit_mask == 1)[0]))
            
            linear_states, cnn_states = self.env.get_mix_states(units_pos_list,n_unit,padding)

            vector_actions,mask,log_prob=self.get_sample_actions(cnn_states, linear_states, units_pos_list)
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
                self.exps_list[i].append([cnn_states[i],linear_states[i],vector_actions[i],rs[i][0],mask[i],done,log_prob[i]])
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

        self.cnn_states_list = None
        self.linear_states_list = None
        self.actions_list = None
        self.rewards_list = None
        self.dones_list = None
        self.old_log_probs_list = None
        self.marks_list = None

    def begin_batch_train(self, samples_list: list):    
        s_cnn_states = [np.array([np.array(s[0]) for s in samples]) for samples in samples_list]
        s_linear_states = [np.array([np.array(s[1]) for s in samples]) for samples in samples_list]
        s_actions = [np.array([s[2] for s in samples]) for samples in samples_list]
        s_masks = [np.array([s[4] for s in samples]) for samples in samples_list]
        s_log_probs = [np.array([s[6] for s in samples]) for samples in samples_list]
        
        s_rewards = [np.array([s[3] for s in samples]) for samples in samples_list]
        s_dones = [np.array([s[5] for s in samples]) for samples in samples_list]
        
        self.cnn_states = [torch.Tensor(states).to(self.device) for states in s_cnn_states]
        self.linear_states = [torch.Tensor(states).to(self.device) for states in s_linear_states]
        self.actions = [torch.Tensor(actions).to(self.device) for actions in s_actions]
        self.old_log_probs = [torch.Tensor(log_probs).to(self.device) for log_probs in s_log_probs]
        self.marks = [torch.Tensor(marks).to(self.device) for marks in s_masks]
        self.rewards = s_rewards
        self.dones = s_dones
        
        self.cnn_states_list = torch.cat([states[0:self.pae_length] for states in self.cnn_states])
        self.linear_states_list = torch.cat([states[0:self.pae_length] for states in self.linear_states])
        self.actions_list = torch.cat([actions[0:self.pae_length] for actions in self.actions])
        self.old_log_probs_list = torch.cat([old_log_probs[0:self.pae_length] for old_log_probs in self.old_log_probs])
        self.marks_list = torch.cat([marks[0:self.pae_length] for marks in self.marks])

    def calculate_samples_gae(self):
        np_advantages = []
        np_returns = []
        
        for cnn_states,linear_states,rewards,dones in zip(self.cnn_states,self.linear_states,self.rewards,self.dones):
            with torch.no_grad():
                _,values = self.calculate_net(cnn_states,linear_states)
                            
            advantages,returns = calculate_gae(values.cpu().numpy().reshape(-1),rewards,dones,gamma,gae_lambda)
            np_advantages.extend(advantages[0:self.pae_length])
            np_returns.extend(returns[0:self.pae_length])
            
        np_advantages = np.array(np_advantages)
        np_returns = np.array(np_returns)
        
        return np_advantages, np_returns
        
    def end_batch_train(self):
        self.cnn_states_list = None
        self.linear_states_list = None
        self.actions_list = None
        self.rewards_list = None
        self.dones_list = None
        self.old_log_probs_list = None
        self.marks_list = None

    def get_pg_loss(self,ratio,advantage):      
        clip_coef = clip_range
        max_clip_coef = max_clip_range
        positive = torch.where(ratio >= 1.0 + clip_coef, 0 * advantage,advantage)
        negtive = torch.where(ratio <= 1.0 - clip_coef,0 * advantage,torch.where(ratio >= max_clip_coef, 0 * advantage,advantage))
        return torch.where(advantage>=0,positive,negtive)*ratio
        
    def get_prob_entropy_value(self, cnn_states, linear_states, actions, masks):
        distris,values = self.calculate_net(cnn_states, linear_states)
        action_masks = torch.split(masks, action_space[1:], dim=1)
        distris = [dist.update_masks(mask,device=self.device) for dist,mask in zip(distris,action_masks)]
        log_probs = torch.stack([dist.log_prob(action) for dist,action in zip(distris,actions[1:])])
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
            
            mini_cnn_states = self.cnn_states_list[start_index:end_index]
            mini_linear_states = self.linear_states_list[start_index:end_index]
            mini_actions = self.actions_list[start_index:end_index]
            mini_masks = self.marks_list[start_index:end_index]
            mini_old_log_probs = self.old_log_probs_list[start_index:end_index]
            
            self.calculate_net.load_state_dict(self.net.state_dict())
                
            mini_new_log_probs,mini_entropys,mini_new_values = self.get_prob_entropy_value(mini_cnn_states,mini_linear_states,mini_actions.T,mini_masks)
                        
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
    writer = SummaryWriter()
    net = ActorCritic()
    agent = Agent(net)
    calculator = Calculator(net)
    MAX_VERSION = 100000
    REPEAT_TIMES = 10
    for version in range(MAX_VERSION):
        samples_list, infos = agent.sample_env(check=True)
        for (key,value) in infos.items():
                writer.add_scalar(key,value,version)

        print("version:",version,"reward:",infos["mean_rewards"])

        calculator.begin_batch_train(samples_list)
        for _ in range(REPEAT_TIMES):
            calculator.generate_grads()
        calculator.end_batch_train()

        if version % 1000 == 0:
            torch.save(net.state_dict(), 'models\\mix_state\\'+str(version)+'.pth')
            torch.save(net, 'models\\mix_state\\'+str(version)+'.pkl')