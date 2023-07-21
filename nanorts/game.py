from nanorts.units import Unit, UnitType, load_unit_types
from nanorts.player import Player
from nanorts.pos import distance, next_dir_pos, next_attack_pos
from nanorts.action import Action

import xml.etree.ElementTree as ET
import numpy as np
import copy

class Game:
    def __init__(self,path:str,reward_weight:dict) -> None:
        self.map_path = path
        self.reward_weight = reward_weight
        #load unit types from json file
        self.unit_types = load_unit_types() # unit_type_name -> unit_type
        self.units:dict[int,Unit] = dict() # unit_int_pos -> unit
        self.players:dict[int,Player] = dict() # player_id -> player
        self.building_pos:dict[int,int] = dict() # unit_pos -> building_pos
        self.moving_pos:dict[int,int] = dict() # unit_pos -> target_pos
        self.width = 0
        self.height = 0
        self.produce_unit_id = 0 # id for next new unit
        self.terrain = None
        # load map from xml file
        self.load_map(self.map_path)
        self.game_time = 0

    def load_map(self, path:str)->None:
        tree = ET.parse(path)
        root = tree.getroot()
        self.width = int(root.get('width'))
        self.height = int(root.get('height'))
        players = root.find('players')
        for player in players.findall('rts.Player'):
            player_id = int(player.get('ID'))
            resources = int(player.get('resources'))
            self.players[player_id] = Player(player_id, resources)
        us = root.find('units')
        for u in us.findall('rts.units.Unit'):
            unit_type = u.get('type')
            player_id = int(u.get('player'))
            x = int(u.get('x'))
            y = int(u.get('y'))
            int_pos = y*self.width+x
            carried_resource = int(u.get('resources'))
            unit_type = self.unit_types[unit_type]
            unit = Unit(self.produce_unit_id, player_id, int_pos, self.width, unit_type, carried_resource)
            self.units[int_pos] = unit
            self.produce_unit_id += 1
        terrain = root.find('terrain').text
        self.terrain = np.zeros((self.height, self.width), dtype=np.int32)
        for i in range(self.height):
            for j in range(self.width):
                t = int(terrain[i*self.width+j])
                if t == 1:
                    self.terrain[i,j] = 1
                    int_pos = i*self.width+j
                    unit_type = self.unit_types['Terrain']
                    self.units[int_pos] = Unit(self.produce_unit_id, -1, int_pos, self.width, unit_type)
                    self.produce_unit_id += 1

    def get_cant_move_pos_vector(self)->np.ndarray:
        res = np.zeros(self.height*self.width, dtype=np.int32)
        for pos in list(self.units.keys()):
            if not self.units[pos].unit_type.canMove:
                res[pos] = 1
        return res
    
    def get_unit_pos_vector(self)->np.ndarray:
        res = np.zeros(self.height*self.width, dtype=np.int32)
        for pos in list(self.units.keys()):
            res[pos] = 1
        return res

    """
    #0: hp 0
    #1: hp 1
    #2: hp 2
    #3: hp 3
    #4: hp > 3
    #5 resource 0
    #6 resource 1
    #7 resource 2
    #8 resource 3
    #9 resource > 3
    #10: player -1
    #11: player 0
    #12: player 1
    #13: no_unit
    #14: resource
    #15: base
    #16: barracks
    #17: worker
    #18: light
    #19: heavy
    #20: ranged
    #21: no_action
    #22: move
    #23: attack
    #24: harvest
    #25: return
    #26: produce"""
    def unit_to_vector(self, unit:Unit)->np.ndarray:
        vector = np.zeros(27)
        if unit.current_hp == 0:
            vector[0] = 1
        elif unit.current_hp == 1:
            vector[1] = 1
        elif unit.current_hp == 2:
            vector[2] = 1
        elif unit.current_hp == 3:
            vector[3] = 1
        else:
            vector[4] = 1
        if unit.carried_resource == 0:
            vector[5] = 1
        elif unit.carried_resource == 1:
            vector[6] = 1
        elif unit.carried_resource == 2:
            vector[7] = 1
        elif unit.carried_resource == 3:
            vector[8] = 1
        else:
            vector[9] = 1
        if unit.player_id == -1:
            vector[10] = 1
        elif unit.player_id == 0:
            vector[11] = 1
        else:
            vector[12] = 1
        if unit.unit_type.isResource:
            vector[14] = 1
        elif unit.unit_type.isStockpile:
            vector[15] = 1
        elif unit.unit_type.name == 'Barracks':
            vector[16] = 1
        elif unit.unit_type.name == 'Worker':
            vector[17] = 1
        elif unit.unit_type.name == 'Light':
            vector[18] = 1
        elif unit.unit_type.name == 'Heavy':
            vector[19] = 1
        elif unit.unit_type.name == 'Ranged':
            vector[20] = 1
        if unit.current_action is None:
            vector[21] = 1
        elif unit.current_action == 'move':
            vector[22] = 1
        elif unit.current_action == 'attack':
            vector[23] = 1
        elif unit.current_action == 'harvest':
            vector[24] = 1
        elif unit.current_action == 'return':
            vector[25] = 1
        elif unit.current_action == 'produce':
            vector[26] = 1
        return vector

    def get_grid_state(self)->np.ndarray:
        grid_state = np.zeros((self.height, self.width, 27), dtype=np.int32)
        for unit in self.units.values():
            x=unit.pos % self.width
            y=unit.pos // self.width
            grid_state[y,x,:] = self.unit_to_vector(unit)
        return grid_state
    
    def get_obstacles(self)->set:
        obstacles = set()
        for unit in list(self.units.values()):
            obstacle = unit.pos
            obstacles.add(obstacle)
        for building_pos in list(self.building_pos.values()):
            obstacle = building_pos
            obstacles.add(obstacle)
        for moving_pos in list(self.moving_pos.keys()):
            obstacle = moving_pos
            obstacles.add(obstacle)
        return obstacles
    
    def stop_unit_action(self, unit_pos:int) -> None:
        if unit_pos not in list(self.units.keys()):
            return
        self.units[unit_pos].current_action = None
        self.units[unit_pos].current_action_target = -1
        self.units[unit_pos].execute_current_action_time = 0
        self.units[unit_pos].building_unit_type = None

    def can_move(self, unit_pos:int, target_pos:int) -> bool:
        if unit_pos not in list(self.units.keys()):
            return False
        if self.units[unit_pos].unit_type.canMove == False:
            return False
        if distance(unit_pos, target_pos, self.width) != 1:
            return False
        if target_pos < 0 or target_pos >= self.width*self.height:
            return False
        if target_pos in list(self.building_pos.values()):
            return False
        if target_pos in list(self.units.keys()):
            return False
        if distance(unit_pos, target_pos, self.width) != 1:
            return False
        return True
    
    def can_harvest(self, unit_pos:int, target_pos:int) -> bool:
        if unit_pos not in list(self.units.keys()):
            return False
        if target_pos not in list(self.units.keys()):
            return False
        unit:Unit = self.units[unit_pos]
        target:Unit = self.units[target_pos]
        if unit.carried_resource > 0:
            return False
        if not unit.unit_type.canHarvest:
            return False
        if not target.unit_type.isResource:
            return False
        if distance(unit_pos, target_pos, self.width) != 1:
            return False
        return True
    
    def can_return(self, unit_pos:int, target_pos:int) -> bool:
        if unit_pos not in list(self.units.keys()):
            return False
        if target_pos not in list(self.units.keys()):
            return False
        unit:Unit = self.units[unit_pos]
        target:Unit = self.units[target_pos]
        if unit.player_id != target.player_id:
            return False
        if unit.carried_resource == 0:
            return False
        if not unit.unit_type.canHarvest:
            return False
        if not target.unit_type.isStockpile:
            return False
        if distance(unit_pos, target_pos, self.width) != 1:
            return False
        return True
    
    def can_produce(self, unit_pos:int, target_pos:int, unit_type_name:str) -> bool:
        if unit_pos not in list(self.units.keys()):
            return False
        if distance(unit_pos, target_pos, self.width) != 1:
            return False
        if target_pos < 0 or target_pos >= self.width*self.height:
            return False
        if target_pos in list(self.units.keys()):
            return False
        if target_pos in list(self.moving_pos.values()):
            return False
        unit:Unit = self.units[unit_pos]
        target_unit_type:UnitType = self.unit_types[unit_type_name]
        for u_pos in list(self.building_pos.keys()):
            if target_pos == self.building_pos[u_pos]:
                if u_pos != unit_pos:
                    return False
        if len(unit.unit_type.produces) < 1:
            return False
        if target_unit_type.name not in unit.unit_type.produces:
            return False
        return True
    
    def can_attack(self, unit_pos:int, target_pos:int) -> bool:
        if unit_pos not in list(self.units.keys()):
            return False
        if target_pos not in list(self.units.keys()):
            return False
        unit:Unit = self.units[unit_pos]
        target:Unit = self.units[target_pos]
        if not unit.unit_type.canAttack:
            return False
        if unit.player_id == target.player_id:
            return False
        if target.player_id == -1:
            return False
        if distance(unit_pos, target_pos, self.width) > unit.unit_type.attackRange:
            return False
        return True

    def begin_move(self, unit_pos:int, target_pos:int) -> None:
        if unit_pos not in list(self.units.keys()):
            return
        if target_pos < 0 or target_pos >= self.width*self.height:
            return
        if not self.units[unit_pos].unit_type.canMove:
            return
        self.units[unit_pos].current_action = 'move'
        self.units[unit_pos].current_action_target = target_pos
        self.units[unit_pos].execute_current_action_time = self.units[unit_pos].unit_type.moveTime

    def begin_harvest(self, unit_pos:int, target_pos:int) -> None:
        if unit_pos not in list(self.units.keys()):
            return
        self.units[unit_pos].current_action = 'harvest'
        self.units[unit_pos].current_action_target = target_pos
        self.units[unit_pos].execute_current_action_time = self.units[unit_pos].unit_type.harvestTime

    def begin_return(self, unit_pos:int, target_pos:int) -> None:
        if unit_pos not in list(self.units.keys()):
            return
        self.units[unit_pos].current_action = 'return'
        self.units[unit_pos].current_action_target = target_pos
        self.units[unit_pos].execute_current_action_time = self.units[unit_pos].unit_type.returnTime

    def begin_produce(self, unit_pos:int, target_pos:int, unit_type_name:str) -> None:
        if unit_pos not in list(self.units.keys()):
            return
        if self.players[self.units[unit_pos].player_id].resource < self.unit_types[unit_type_name].cost:
            return
        self.units[unit_pos].current_action = 'produce'
        self.units[unit_pos].current_action_target = target_pos
        self.units[unit_pos].execute_current_action_time = self.unit_types[unit_type_name].produceTime
        self.units[unit_pos].building_unit_type = self.unit_types[unit_type_name]
        self.players[self.units[unit_pos].player_id].resource -= self.units[unit_pos].building_unit_type.cost

    def begin_attack(self, unit_pos:int, target_pos:int) -> None:
        if unit_pos not in list(self.units.keys()):
            return
        if target_pos < 0 or target_pos >= self.width*self.height:
            return
        if distance(unit_pos, target_pos, self.width) > self.units[unit_pos].unit_type.attackRange:
            return
        self.units[unit_pos].current_action = 'attack'
        self.units[unit_pos].current_action_target = target_pos
        self.units[unit_pos].execute_current_action_time = self.units[unit_pos].unit_type.attackTime

    def set_ocuppied_pos(self) -> None:
        self.building_pos = dict()
        self.moving_pos = dict()
        for unit in self.units.values():
            if unit.current_action == 'move':
                self.moving_pos[unit.pos] = unit.current_action_target
            elif unit.current_action == 'produce':
                self.building_pos[unit.pos] = unit.current_action_target

    def execute_unit_action(self,unit_pos:int):
        if unit_pos not in list(self.units.keys()):
            return (0,0)
        unit:Unit = self.units[unit_pos]
        if unit.current_action is None:
            return (0,0)
        if unit.current_action == 'move':
            rewards = self.execute_move_unit(unit_pos)
        elif unit.current_action == 'harvest':
            rewards = self.execute_harvest_unit(unit_pos)
        elif unit.current_action == 'return':
            rewards = self.execute_return_unit(unit_pos)
        elif unit.current_action == 'produce':
            rewards = self.execute_produce_unit(unit_pos)
        elif unit.current_action == 'attack':
            rewards = self.execute_attack_unit(unit_pos)
        self.set_ocuppied_pos()
        return rewards
    
    def execute_move_unit(self, unit_pos:int):
        if unit_pos not in list(self.units.keys()):
            return (0,0)
        if self.units[unit_pos].current_action != 'move':
            self.stop_unit_action(unit_pos)
            return (0,0)
        if self.units[unit_pos].execute_current_action_time > 0:
            self.units[unit_pos].execute_current_action_time -= 1
            return (0,0)
        target_pos = self.units[unit_pos].current_action_target
        if not self.can_move(unit_pos, target_pos):
            self.stop_unit_action(unit_pos)
            return (0,0)
        unit = self.units[unit_pos]
        self.units[target_pos] = copy.deepcopy(unit)
        self.units.pop(unit_pos)
        self.units[target_pos].pos = target_pos
        self.stop_unit_action(unit_pos)
        return (0,0)

    def execute_harvest_unit(self, unit_pos:int):
        if unit_pos not in list(self.units.keys()):
            return (0,0)
        if self.units[unit_pos].current_action != 'harvest':
            self.stop_unit_action(unit_pos)
            return (0,0)
        if self.units[unit_pos].execute_current_action_time > 0:
            self.units[unit_pos].execute_current_action_time -= 1
            return (0,0)
        target_pos = self.units[unit_pos].current_action_target
        if not self.can_harvest(unit_pos, target_pos):
            self.stop_unit_action(unit_pos)
            return (0,0)
        self.units[unit_pos].carried_resource = self.units[unit_pos].unit_type.harvestAmount
        self.units[target_pos].carried_resource -= self.units[unit_pos].unit_type.harvestAmount
        if self.units[target_pos].carried_resource <= 0:
            self.units.pop(target_pos)
        self.stop_unit_action(unit_pos)
        if self.units[unit_pos].player_id == 0:
            return (self.reward_weight['harvest'],0)
        else:
            return (0,self.reward_weight['harvest'])

    def execute_return_unit(self, unit_pos:int):
        if unit_pos not in list(self.units.keys()):
            return (0,0)
        if self.units[unit_pos].current_action != 'return':
            self.stop_unit_action(unit_pos)
            return (0,0)
        if self.units[unit_pos].execute_current_action_time > 0:
            self.units[unit_pos].execute_current_action_time -= 1
            return (0,0)
        target_pos = self.units[unit_pos].current_action_target
        if not self.can_return(unit_pos, target_pos):
            self.stop_unit_action(unit_pos)
            return (0,0)
        self.players[self.units[unit_pos].player_id].resource += self.units[unit_pos].carried_resource
        self.units[unit_pos].carried_resource = 0
        self.stop_unit_action(unit_pos)
        if self.units[unit_pos].player_id == 0:
            return (self.reward_weight['return'],0)
        else:
            return (0,self.reward_weight['return'])

    def execute_produce_unit(self, unit_pos:int):
        if unit_pos not in list(self.units.keys()):
            return (0,0)
        if self.units[unit_pos].current_action != 'produce':
            self.players[self.units[unit_pos].player_id].resource += self.units[unit_pos].building_unit_type.cost
            self.stop_unit_action(unit_pos)
            return (0,0)
        if self.units[unit_pos].execute_current_action_time > 0:
            self.units[unit_pos].execute_current_action_time -= 1
            return (0,0)
        target_pos = self.units[unit_pos].current_action_target
        if not self.can_produce(unit_pos, target_pos, self.units[unit_pos].building_unit_type.name):
            #self.players[self.units[unit_pos].player_id].resource += self.units[unit_pos].building_unit_type.cost
            self.stop_unit_action(unit_pos)
            return (0,0)
        prodeced_unit = Unit(self.produce_unit_id, self.units[unit_pos].player_id, target_pos, self.width, self.units[unit_pos].building_unit_type)
        self.units[target_pos] = prodeced_unit
        self.produce_unit_id += 1
        
        self.stop_unit_action(unit_pos)
        if self.units[unit_pos].player_id == 0:
            return (self.reward_weight['produce'],0)
        else:
            return (0,self.reward_weight['produce'])

    def execute_attack_unit(self, unit_pos:int):
        if unit_pos not in list(self.units.keys()):
            return (0,0)
        if self.units[unit_pos].current_action != 'attack':
            return (0,0)
        if self.units[unit_pos].execute_current_action_time > 0:
            self.units[unit_pos].execute_current_action_time -= 1
            return (0,0)
        target_pos = self.units[unit_pos].current_action_target
        if not self.can_attack(unit_pos, target_pos):
            self.stop_unit_action(unit_pos)
            return (0,0)
        if self.units[unit_pos].unit_type.minDamage == self.units[unit_pos].unit_type.maxDamage:
            damage = self.units[unit_pos].unit_type.minDamage
        else:
            damage = np.random.randint(self.units[unit_pos].unit_type.minDamage, self.units[unit_pos].unit_type.maxDamage+1)
        self.units[target_pos].current_hp -= damage
        if self.units[target_pos].current_hp <= 0:
            if self.units[target_pos].current_action == 'produce':
                self.players[self.units[target_pos].player_id].resource += self.units[target_pos].building_unit_type.cost
            self.units.pop(target_pos)
        self.stop_unit_action(unit_pos)
        if self.units[unit_pos].player_id == 0:
            return (self.reward_weight['attack'],0)
        else:
            return (0,self.reward_weight['attack'])

    def run(self):
        self.game_time += 1
        r0 = 0
        r1 = 0
        done = False
        winner = -1
        n_player0_units = 0
        n_player1_units = 0
        all_units_pos = copy.deepcopy(list(self.units.keys()))
        for unit_pos in all_units_pos:
            if unit_pos not in list(self.units.keys()):
                continue
            if self.units[unit_pos].player_id == 0:
                n_player0_units += 1
            elif self.units[unit_pos].player_id == 1:
                n_player1_units += 1
            r0_,r1_ = self.execute_unit_action(unit_pos)
            r0 += r0_
            r1 += r1_
        if n_player0_units == 0:
            done = True
            winner = 1
        elif n_player1_units == 0:
            done = True
            winner = 0
        return (r0,r1), done, winner
    
    def get_player_available_units(self, player_id:int)->list:
        available_units = []
        for unit_pos in list(self.units.keys()):
            if self.units[unit_pos].player_id == player_id and not self.units[unit_pos].busy():
                available_units.append(unit_pos)
        return available_units
    
    def get_player_available_actions(self, player_id:int)->list:
        self.set_ocuppied_pos()
        available_actions = []
        available_units = self.get_player_available_units(player_id)
        if len(available_units) < 1:
            return available_actions
        for unit_pos in available_units:
            unit:Unit = self.units[unit_pos]
            if unit.unit_type.canMove:
                for dir in range(4):
                    target_pos = next_dir_pos(unit_pos, dir, self.width)
                    if self.can_move(unit_pos, target_pos):
                        action = Action(unit_pos, 'move', target_pos)
                        available_actions.append(action)
            if unit.unit_type.canHarvest:
                for dir in range(4):
                    target_pos = next_dir_pos(unit_pos, dir, self.width)
                    if self.can_harvest(unit_pos, target_pos):
                        action = Action(unit_pos, 'harvest', target_pos)
                        available_actions.append(action)
                    if self.can_return(unit_pos, target_pos):
                        action = Action(unit_pos, 'return', target_pos)
                        available_actions.append(action)
            if len(unit.unit_type.produces) > 0:
                for dir in range(4):
                    target_pos = next_dir_pos(unit_pos, dir, self.width)
                    for produce_type in unit.unit_type.produces:
                        if self.can_produce(unit_pos, target_pos, produce_type):
                            action = Action(unit_pos, 'produce', target_pos, produce_type)
                            available_actions.append(action)
            if unit.unit_type.canAttack:
                attack_range = unit.unit_type.attackRange
                for i in range(-attack_range, attack_range+1):
                    for j in range(-attack_range, attack_range+1):
                        target_pos_x = unit.pos % self.width + i
                        target_pos_y = unit.pos // self.width + j
                        target_pos = target_pos_y * self.width + target_pos_x
                        if self.can_attack(unit_pos, target_pos):
                            action = Action(unit_pos, 'attack', target_pos)
                            available_actions.append(action)
        return available_actions
    
    def reset(self):
        self.unit_types = load_unit_types() # unit_type_name -> unit_type
        self.units = dict() # unit_int_pos -> unit
        self.players = dict() # player_id -> player
        self.building_pos = dict()
        self.moving_pos = dict()
        self.width = 0
        self.height = 0
        self.produce_unit_id = 0 # id for next new unit
        self.terrain = None
        self.load_map(self.map_path)
        self.game_time = 0
        return self.get_grid_state()
    
    def get_vector_units_mask(self, player_id:int)->np.ndarray:
        units_mask = np.zeros(self.width*self.height)
        for unit_pos in list(self.units.keys()):
            if self.units[unit_pos].player_id == player_id and not self.units[unit_pos].busy():
                units_mask[unit_pos] = 1
        return units_mask
    
    #[0:6] NOOP, move, harvest, return, produce, attack
    #[6:10] move: up, right, down, left
    #[10:14] harvest: up, right, down, left
    #[14:18] return: up, right, down, left
    #[18:22] produce: up, right, down, left
    #[22:29] produce_type: resource, base, barracks, worker, light, heavy, ranged
    #[29:78] attack_pos: 1~7*7
    def get_vector_action_mask(self, unit_pos:int, player_id:int)->np.ndarray:
        if unit_pos not in list(self.units.keys()):
            return np.zeros(78)
        unit:Unit = self.units[unit_pos]
        if unit.player_id != player_id:
            return np.zeros(78)
        action_mask = np.zeros(78)
        nextpos = [unit.pos-self.width, unit.pos+1, unit.pos+self.width, unit.pos-1]
        for i in range(4):
            target_pos = nextpos[i]
            if target_pos < 0 or target_pos >= self.width * self.height:
                continue
            if distance(unit.pos, target_pos, self.width) != 1:
                continue
            in_building = False
            for u_pos in list(self.building_pos.keys()):
                if self.building_pos[u_pos] == target_pos:
                    if self.building_pos[u_pos] != unit.current_action_target:
                        in_building = True
                        break
            if in_building:
                continue
            if target_pos not in list(self.units.keys()):
                if unit.unit_type.canMove:
                    action_mask[1] = 1
                    action_mask[6+i] = 1
                if len(unit.unit_type.produces) > 0:
                    action_mask[4] = 1
                    action_mask[18+i] = 1
                    for produce_type in unit.unit_type.produces:
                        if self.unit_types[produce_type].cost > self.players[unit.player_id].resource:
                            continue
                        if produce_type == 'Resource':
                            action_mask[22] = 1
                        elif produce_type == 'Base':
                            action_mask[23] = 1
                        elif produce_type == 'Barracks':
                            action_mask[24] = 1
                        elif produce_type == 'Worker':
                            action_mask[25] = 1
                        elif produce_type == 'Light':
                            action_mask[26] = 1
                        elif produce_type == 'Heavy':
                            action_mask[27] = 1
                        elif produce_type == 'Ranged':
                            action_mask[28] = 1
            else:
                if target_pos not in list(self.units.keys()):
                    continue
                target_unit:Unit = self.units[target_pos]
                if target_unit.unit_type.name == 'Resource' and unit.carried_resource < 1 and unit.unit_type.canHarvest:
                    action_mask[2] = 1
                    action_mask[10+i] = 1
                if target_unit.unit_type.isStockpile and unit.carried_resource > 0 and unit.unit_type.canHarvest:
                    action_mask[3] = 1
                    action_mask[14+i] = 1
        if unit.unit_type.canAttack:
            for i in range(-unit.unit_type.attackRange, unit.unit_type.attackRange+1):
                for j in range(-unit.unit_type.attackRange, unit.unit_type.attackRange+1):
                    if abs(i) + abs(j) > unit.unit_type.attackRange:
                        continue
                    d = j+3+(i+3)*7
                    next_pos = next_attack_pos(unit.pos, d, self.width)
                    if distance(unit.pos, next_pos, self.width) > unit.unit_type.attackRange:
                        continue
                    if next_pos not in list(self.units.keys()):
                        continue
                    target_unit = self.units[next_pos]
                    if target_unit.player_id == unit.player_id or target_unit.player_id == -1:
                        continue
                    action_mask[5] = 1
                    action_mask[29+d] = 1
        return action_mask
    
    #1 NOOP, move, harvest, return, produce, attack
    #2 move: up, right, down, left
    #3 harvest: up, right, down, left
    #4 return: up, right, down, left
    #5 produce: up, right, down, left
    #6 produce_type: resource, base, barracks, worker, light, heavy, ranged
    #7 attack_pos: 1~7*7
    def vector_to_action(self, vector):
        unit_pos = vector[0]
        next_pos = [unit_pos-self.width, unit_pos+1, unit_pos+self.width, unit_pos-1]
        dir = None
        produce_type = None
        target_pos = None
        if vector[1] == 1:
            unit_action_type = 'move'
            target_pos = next_pos[vector[2]]
        elif vector[1] == 2:
            unit_action_type = 'harvest'
            target_pos = next_pos[vector[3]]
        elif vector[1] == 3:
            unit_action_type = 'return'
            target_pos = next_pos[vector[4]]
        elif vector[1] == 4:
            unit_action_type = 'produce'
            target_pos = next_pos[vector[5]]
            if vector[6] == 0:
                produce_type = 'Resource'
            elif vector[6] == 1:
                produce_type = 'Base'
            elif vector[6] == 2:
                produce_type = 'Barracks'
            elif vector[6] == 3:
                produce_type = 'Worker'
            elif vector[6] == 4:
                produce_type = 'Light'
            elif vector[6] == 5:
                produce_type = 'Heavy'
            elif vector[6] == 6:
                produce_type = 'Ranged'
        elif vector[1] == 5:
            unit_action_type = 'attack'
        else:
            unit_action_type = 'NOOP'

        if unit_action_type == 'attack':
            atk_x = vector[7] % 7 - 3
            atk_y = vector[7] // 7 - 3
            unit_pos_x = unit_pos % self.width
            unit_pos_y = unit_pos // self.width
            target_pos = (unit_pos_x + atk_x) + (unit_pos_y + atk_y) * self.width
        return Action(unit_pos, unit_action_type, target_pos, produce_type)



            