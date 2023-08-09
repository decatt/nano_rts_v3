from mcts.rollout import Rollout
from mcts.mctsnode import MCTSNode
from nanorts.game import Game

import numpy as np

def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs

class MCTS:
    def __init__(self, ai1, ai2, c_puct=5, n_playout=1000):
        self.ai1 = ai1
        self.ai2 = ai2
        self.c_puct = c_puct
        self.n_playout = n_playout
        self.root = MCTSNode(None, 1.0)
    
    def get_move_probs(self, game:Game, temp=1e-3):
        for n in range(self.n_playout):
            node = self.root
            game_simulate = game.deepcopy()
            if node.is_leaf():
                break
            action, node = node.select(self.c_puct)
            game_simulate.step(action)
            action_probs = self.ai1.get_action_probs(game_simulate)
            end, winner = game_simulate.run()
            if not end:
                node.expand(action_probs)
            leaf_value = self.evaluate_rollout(game_simulate)
            node.update_recursive(-leaf_value)
        act_visits = [(act, node._n_visits) for act, node in self.root._children.items()]
        acts, visits = zip(*act_visits)
        act_probs = softmax(1.0/temp * np.log(np.array(visits) + 1e-10))
        return acts, act_probs
    
    def evaluate_rollout(self, game):
        ai1 = self.ai1
        ai2 = self.ai2
        rollout = Rollout(game, ai1, ai2, ai1.get_action(game))
        reward, winner = rollout.run()
        return reward[0]

    def update_with_move(self, last_move):
        if last_move in self.root._children:
            self.root = self.root._children[last_move]
            self.root._parent = None
        else:
            self.root = MCTSNode(None, 1.0)

    def get_action(self, game:Game, temp=1e-3, return_prob=False):
        acts, probs = self.get_move_probs(game, temp)
        if return_prob:
            return acts, probs
        return acts[np.argmax(probs)]