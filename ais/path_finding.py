from nanorts.pos import distance

class PathFinding:
    def __init__(self, width:int, height:int):
        self.width = width
        self.height = height
    
    def find_path(self, start:int, target:int, obstacles:set, ignore_target:bool = True)->list:
        pass

    def get_move_pos(self, start:int, target:int, obstacles:set, max_range=6, ignore_target:bool = True)->int:
        pass

    def next_pos(self, current:int, dir:int)->int:
        if dir == 0:
            return current - self.width
        elif dir == 1:
            return current + 1
        elif dir == 2:
            return current + self.width
        elif dir == 3:
            return current - 1

    def get_distance(self, pos1:int, pos2:int)->int:
        pos1_x = pos1 % self.width
        pos1_y = pos1 // self.width
        pos2_x = pos2 % self.width
        pos2_y = pos2 // self.width
        return abs(pos1_x - pos2_x) + abs(pos1_y - pos2_y)
    
class BFS(PathFinding):
    def __init__(self, width:int, height:int):
        super().__init__(width, height)
    
    def find_path(self, start:int, target:int, obstacles:set, ignore_target:bool = True)->list:
        if ignore_target:
            if target in obstacles:
                obstacles.remove(target)
        if start == target:
            return []
        queue = [start]
        visited = set()
        visited.add(start)
        parent = {}
        while len(queue) > 0:
            current = queue.pop(0)
            if current == target:
                break
            for dir in range(4):
                next_pos = self.next_pos(current, dir)
                if distance(current, next_pos, self.width) != 1:
                    continue
                if next_pos < 0 or next_pos >= self.width*self.height:
                    continue
                if next_pos in visited or next_pos in obstacles:
                    continue
                visited.add(next_pos)
                parent[next_pos] = current
                queue.append(next_pos)
        if target not in parent:
            return []
        path = []
        current = target
        while current != start:
            path.append(current)
            current = parent[current]
        path.reverse()
        return path
    
    def get_move_pos(self, start:int, target:int, obstacles:set, max_range=6, ignore_target:bool = True)->int:
        if self.get_distance(start, target) <= max_range:
            path = self.find_path(start, target, obstacles, ignore_target)
            if len(path) > 0:
                return path[0]
            
        move_pos = start
        lowest_distance = 999999
        for possible_move_pos in [start-self.width, start+1, start+self.width, start-1, start]:
            if possible_move_pos<0 or possible_move_pos>=self.width*self.height:
                continue
            if distance(start, possible_move_pos, self.width) != 1:
                continue
            if possible_move_pos in obstacles:
                continue
            if self.get_distance(possible_move_pos, target) < lowest_distance:
                lowest_distance = self.get_distance(possible_move_pos, target)
                move_pos = possible_move_pos
        return move_pos

