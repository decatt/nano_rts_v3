from nanorts.game import Game
from nanorts.units import Unit
from nanorts.action import Action
from ais.nano_rts_ai import AI, RushAI
import numpy as np
from nanorts.render import Render
import torch

class GameEnv:
    def __init__(self, map_paths, reward_wrights, max_steps, skip_frames=5, if_render=True):
        self.games:list[Game] = []
        self.map_paths = map_paths
        self.reward_wrights = reward_wrights
        self.max_steps = max_steps
        self.num_envs = len(map_paths)
        for map_path in map_paths:
            self.games.append(Game(map_path, reward_wrights))
        self.num_envs = len(self.games)
        self.skip_frames = skip_frames
        self.action_lists_record = dict()
        if if_render:
            h = self.games[0].height
            w = self.games[0].width
            self.rendering = Render(h,w)

    def step(self, actions0:list[Action], actions1:list[Action])->tuple[list[np.ndarray], list[int], list[bool], list]:
        states = []
        rewards = []
        dones = []
        winners = []
        for i in range(len(self.games)):
            game:Game = self.games[i]
            action0:Action = actions0[i]
            action1:Action = actions1[i]
            for player_id in [0,1]:
                game.set_ocuppied_pos()
                action = action0 if player_id == 0 else action1
                unit_pos = action.unit_pos
                if unit_pos is None:
                    continue
                action_type = action.action_type
                target_pos = action.target_pos
                produced_unit_type = action.produced_unit_type
                if unit_pos not in list(game.units.keys()):
                    continue
                unit:Unit = game.units[unit_pos]
                if unit.player_id != player_id:
                    continue
                if action_type == 'move':
                    game.begin_move(unit_pos, target_pos)
                elif action_type == 'harvest':
                    game.begin_harvest(unit_pos, target_pos)
                elif action_type == 'return':
                    game.begin_return(unit_pos, target_pos)
                elif action_type == 'produce':
                    game.begin_produce(unit_pos, target_pos, produced_unit_type)
                elif action_type == 'attack':
                    game.begin_attack(unit_pos, target_pos)
            reward,done,winner = game.run()
            state = game.get_grid_state()

            if done:
                state = game.reset()
            if game.game_time >= self.max_steps:
                done = True
                state = game.reset()
            states.append(state)
            rewards.append(reward)
            dones.append(done)
            winners.append(winner)
        return states, rewards, dones, winners
    
    def one_game_step(self, action0:Action, action1:Action)->tuple[np.ndarray, int, bool, list]:
        game:Game = self.games[0]
        for player_id in [0,1]:
            game.set_ocuppied_pos()
            action = action0 if player_id == 0 else action1
            unit_pos = action.unit_pos
            if unit_pos is None:
                continue
            action_type = action.action_type
            target_pos = action.target_pos
            produced_unit_type = action.produced_unit_type
            if unit_pos not in list(game.units.keys()):
                continue
            unit:Unit = game.units[unit_pos]
            if unit.player_id != player_id:
                continue
            if action_type == 'move':
                game.begin_move(unit_pos, target_pos)
            elif action_type == 'harvest':
                game.begin_harvest(unit_pos, target_pos)
            elif action_type == 'return':
                game.begin_return(unit_pos, target_pos)
            elif action_type == 'produce':
                game.begin_produce(unit_pos, target_pos, produced_unit_type)
            elif action_type == 'attack':
                game.begin_attack(unit_pos, target_pos)
        reward,done,winner = game.run()
        state = game.get_grid_state()
        if done:
            state = game.reset()
        if game.game_time >= self.max_steps:
            done = True
            state = game.reset()
        return state, reward, done, winner

    def step_action_lists(self, action_lists:list[list[Action]])->tuple[list[np.ndarray], list[int], list[bool], list]:
        states = []
        rewards = []
        dones = []
        winners = []
        for i in range(len(self.games)):
            game:Game = self.games[i]
            if game.game_time % self.skip_frames == 0:
                game.run()
            action_list = action_lists[i]
            for action in action_list:
                game.set_ocuppied_pos()
                unit_pos = action.unit_pos
                if unit_pos is None:
                    continue
                action_type = action.action_type
                target_pos = action.target_pos
                produced_unit_type = action.produced_unit_type
                if unit_pos not in list(game.units.keys()):
                    continue
                if action_type == 'move':
                    game.begin_move(unit_pos, target_pos)
                elif action_type == 'harvest':
                    game.begin_harvest(unit_pos, target_pos)
                elif action_type == 'return':
                    game.begin_return(unit_pos, target_pos)
                elif action_type == 'produce':
                    game.begin_produce(unit_pos, target_pos, produced_unit_type)
                elif action_type == 'attack':
                    game.begin_attack(unit_pos, target_pos)
            reward,done,winner = game.run()
            state = game.get_grid_state()
            if done:
                state = game.reset()
            if game.game_time >= self.max_steps:
                done = True
                state = game.reset()
            states.append(state)
            rewards.append(reward)
            dones.append(done)
            winners.append(winner)
        return states, rewards, dones, winners

    def reset(self)->np.ndarray:
        states = []
        for i in range(len(self.games)):
            state = self.games[i].reset()
            states.append(state)
        return np.array(states)
    
    def render(self)->None:
        self.rendering.draw(self.games[0])
        
    def get_unit_masks(self,player_id:int)->np.ndarray:
        unit_masks = []
        for i in range(len(self.games)):
            game:Game = self.games[i]
            unit_mask = game.get_vector_units_mask(player_id)
            unit_masks.append(unit_mask)
        return np.array(unit_masks)

    def get_action_masks(self, units, player_id=0)->np.ndarray:
        action_masks = []
        for i in range(len(self.games)):
            game:Game = self.games[i]
            action_mask = game.get_vector_action_mask(units[i], player_id)
            action_masks.append(action_mask)
        return np.array(action_masks)
    
    def get_mix_states(self, unit_pos_list:list[int], n_units:int, paddind:int):
        n_games = len(self.games)
        res_cnn_states = torch.zeros((n_games, 2*paddind+1, 2*paddind+1, 27))
        res_linear_states = torch.zeros((n_games, n_units, 29))
        for i in range(len(self.games)):
            unit_pos = unit_pos_list[i]
            linear_state, cnn_state = self.games[i].get_mix_state(unit_pos, n_units, paddind)
            res_cnn_states[i] = torch.Tensor(cnn_state)
            res_linear_states[i] = torch.Tensor(linear_state)
        return res_linear_states, res_cnn_states