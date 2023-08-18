from mcts.rollout import Rollout
from nanorts.game import Game
from nanorts.action import Action

class MCAI:
    def __init__(self, ai1, ai2, player_id, num_playout=10, max_steps=100, discount=0.99):
        self.ai1 = ai1
        self.ai2 = ai2
        self.player_id = player_id
        self.num_playout = num_playout
        self.max_steps = max_steps
        self.discount = discount

    def get_action(self, game:Game):
        play_out_game = game.deepcopy()
        available_actions = game.get_player_available_actions(self.player_id)
        if len(available_actions) == 0:
            return Action(None, None, None, None)
        max_reward = -1e9
        best_action = Action(None, None, None, None)
        for action in available_actions:
            tot_reward = 0
            for _ in range(self.num_playout):
                playout = Rollout(play_out_game, self.ai1, self.ai2, action, max_steps=self.max_steps, player_id=self.player_id, discount=self.discount)
                reward, winner = playout.run()
                tot_reward += reward
            if tot_reward > max_reward:
                max_reward = tot_reward
                best_action = action
        return best_action