from nanorts.units import Unit, UnitType
from numpy import ndarray
import numpy as np

class Action:
    def __init__(self, unit_pos:int, action_type:str, target_pos:int, produced_unit_type:str=None):
        self.unit_pos = unit_pos
        self.action_type = action_type
        self.target_pos = target_pos
        self.produced_unit_type = produced_unit_type

    def action_to_vector(self, map_width:int)->ndarray:
        vector = np.zeros(8)
        if self.unit_pos is None:
            return vector
        vector[0] = self.unit_pos
        unit_x = self.unit_pos % map_width
        unit_y = self.unit_pos // map_width
        if self.target_pos is None:
            return vector
        target_x = self.target_pos % map_width
        target_y = self.target_pos // map_width
        dir = (target_x - unit_x, target_y - unit_y)
        d = 0
        if dir == (0,-1):
            d = 0
        elif dir == (1,0):
            d = 1
        elif dir == (0,1):
            d = 2
        elif dir == (-1,0):
            d = 3
        
        if self.action_type == 'move':
            vector[1] = 1
            vector[2] = d
        elif self.action_type == 'harvest':
            vector[1] = 2
            vector[3] = d
        elif self.action_type == 'return':
            vector[1] = 3
            vector[4] = d
        elif self.action_type == 'produce':
            vector[1] = 4
            vector[5] = d
            if self.produced_unit_type == 'Base':
                vector[6] = 1
            elif self.produced_unit_type == 'Barracks':
                vector[6] = 2
            elif self.produced_unit_type == 'Worker':
                vector[6] = 3
            elif self.produced_unit_type == 'Light':
                vector[6] = 4
            elif self.produced_unit_type == 'Heavy':
                vector[6] = 5
            elif self.produced_unit_type == 'Ranged':
                vector[6] = 6
        elif self.action_type == 'attack':
            vector[1] = 5
            vector[7] = dir[0]+3 + 7*(dir[1]+3)
        return vector
    
    def action_to_one_hot(self, map_width:int, map_height:int)->ndarray:
        vector = self.action_to_vector(map_width)
        res = np.zeros((map_width*map_height+6+4+4+4+4+7+49,))
        res[int(vector[0])] = 1
        res[map_width*map_height+int(vector[1])] = 1
        res[map_width*map_height+6+int(vector[2])] = 1
        res[map_width*map_height+6+4+int(vector[3])] = 1
        res[map_width*map_height+6+4+4+int(vector[4])] = 1
        res[map_width*map_height+6+4+4+4+int(vector[5])] = 1
        res[map_width*map_height+6+4+4+4+4+int(vector[6])] = 1
        res[map_width*map_height+6+4+4+4+4+7+int(vector[7])] = 1
        return res
    
    def action_to_one_hot2(self, map_width:int, map_height:int)->ndarray:
        vector = self.action_to_vector(map_width)
        res = np.ones((map_width*map_height+6+4+4+4+4+7+49,))*1e-5
        res[int(vector[0])] = 1
        res[map_width*map_height+int(vector[1])] = 1
        res[map_width*map_height+6+int(vector[2])] = 1
        res[map_width*map_height+6+4+int(vector[3])] = 1
        res[map_width*map_height+6+4+4+int(vector[4])] = 1
        res[map_width*map_height+6+4+4+4+int(vector[5])] = 1
        res[map_width*map_height+6+4+4+4+4+int(vector[6])] = 1
        res[map_width*map_height+6+4+4+4+4+7+int(vector[7])] = 1
        return res

    def action_to_one_hot_w(self, map_width:int, map_height:int, w=-1e2)->ndarray:
        vector = self.action_to_vector(map_width)
        res = np.ones((map_width*map_height+6+4+4+4+4+7+49,))*w
        res[int(vector[0])] = 1
        res[map_width*map_height+int(vector[1])] = 1
        res[map_width*map_height+6+int(vector[2])] = 1
        res[map_width*map_height+6+4+int(vector[3])] = 1
        res[map_width*map_height+6+4+4+int(vector[4])] = 1
        res[map_width*map_height+6+4+4+4+int(vector[5])] = 1
        res[map_width*map_height+6+4+4+4+4+int(vector[6])] = 1
        res[map_width*map_height+6+4+4+4+4+7+int(vector[7])] = 1
        return res
    
    def action_to_array(self, map_width:int, height:int)->ndarray:
        array = np.zeros((height*map_width, 7),dtype = np.int64)
        vector = self.action_to_vector(map_width)
        array[self.unit_pos] = vector[1:]
        return array

    def __str__(self):
        return f'unit_pos: {self.unit_pos}, action_type: {self.action_type}, target_pos: {self.target_pos}, produced_unit_type: {self.produced_unit_type}'
