class Pos:
    def __init__(self, pos:int, width, height):
        self.pos = pos
        self.width = width
        self.height = height
        self.x = self.pos % self.width
        self.y = self.pos // self.width

    def next_move_pos(self, dir:int):
        if dir == 0:
            return Pos(self.pos - self.width, self.width)
        elif dir == 1:
            return Pos(self.pos + 1, self.width)
        elif dir == 2:
            return Pos(self.pos + self.width, self.width)
        elif dir == 3:
            return Pos(self.pos - 1, self.width)
        else:
            return None
    
    def next_attack_pos(self, p:int, range:int=3):
        if p<0 or p>(2*range+1)**2-1:
            return None
        p_x = p % (2*range+1) - range
        p_y = p // (2*range+1) - range
        return Pos(self.pos + p_x + p_y*self.width, self.width)
    
def is_in_range(pos1:int, pos2:int, width:int, range:int)->bool:
    pos1_x = pos1 % width
    pos1_y = pos1 // width
    pos2_x = pos2 % width
    pos2_y = pos2 // width
    return abs(pos1_x-pos2_x) + abs(pos1_y-pos2_y) <= range

def distance(pos1:int, pos2:int, width:int)->int:
    pos1_x = pos1 % width
    pos1_y = pos1 // width
    pos2_x = pos2 % width
    pos2_y = pos2 // width
    return abs(pos1_x-pos2_x) + abs(pos1_y-pos2_y)

def next_dir_pos(pos:int, dir:int, width:int):
    if dir == 0:
        return pos - width
    elif dir == 1:
        return pos + 1
    elif dir == 2:
        return pos + width
    elif dir == 3:
        return pos - 1
    else:
        return -1
    
def next_attack_pos(pos:int, p:int, width:int, range:int=3):
    if p<0 or p>(2*range+1)**2-1:
        return -1
    p_x = p % (2*range+1) - range
    p_y = p // (2*range+1) - range
    return pos + p_x + p_y*width


        
    
