from nanorts.game import Game
from ais.nano_rts_ai import RushAI
from nanorts.action import Action
from nanorts.units import Unit

class Playout:
    def __init__(self, game: Game, ai1: RushAI, ai2: RushAI, next_action: Action, max_steps=1000, player_id=0, discount=0.99):
        self.game = game
        self.ai1 = ai1
        self.ai2 = ai2
        self.next_action = next_action
        self.max_steps = max_steps
        self.player_id = player_id
        self.discount = discount

    def step(self, action1:Action, action2:Action):
        for player_id in [0,1]:
            action = action1 if player_id == 0 else action2
            unit_pos = action.unit_pos
            if unit_pos is None:
                continue
            action_type = action.action_type
            target_pos = action.target_pos
            produced_unit_type = action.produced_unit_type
            if unit_pos not in list(self.game.units.keys()):
                continue
            unit:Unit = self.game.units[unit_pos]
            if unit.player_id != player_id:
                continue
            if action_type == 'move':
                self.game.begin_move(unit_pos, target_pos)
            elif action_type == 'harvest':
                self.game.begin_harvest(unit_pos, target_pos)
            elif action_type == 'return':
                self.game.begin_return(unit_pos, target_pos)
            elif action_type == 'produce':
                self.game.begin_produce(unit_pos, target_pos, produced_unit_type)
            elif action_type == 'attack':
                self.game.begin_attack(unit_pos, target_pos)
        return self.game.run()
    
    def run(self):
        tot_reward = 0
        next_action1 = self.next_action
        next_action2 = self.ai2.get_action(self.game)
        reward,done,winner = self.step(next_action1, next_action2)
        tot_reward += reward[self.player_id]
        if done:
            return tot_reward, winner
        d = self.discount
        for _ in range(self.max_steps):
            next_action1 = self.ai1.get_action(self.game)
            next_action2 = self.ai2.get_action(self.game)
            reward,done,winner = self.step(next_action1, next_action2)
            tot_reward += d*reward[self.player_id]
            d *= self.discount
            if done:
                return tot_reward, winner
        return tot_reward, winner


