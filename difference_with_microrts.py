import numpy as np
import time
import random

from gym_microrts.envs.vec_env import MicroRTSGridModeVecEnv
from nanorts.game_env import GameEnv
from ais.nano_rts_ai import RushAI

seed = 0
np.random.seed(seed)
random.seed(seed)

nano_rts_env = GameEnv(
    ['maps/10x10/basesWorkers10x10.xml' for _ in range(1)], 
    {'win': 10, 'harvest': 1, 'return': 1, 'produce': 0.2, 'attack': 1}, 
    5000
    )

micro_rts_env = MicroRTSGridModeVecEnv(
    num_selfplay_envs=2,
    num_bot_envs=0,
    max_steps=2000,
    render_theme=2,
    ai2s=[],
    map_paths=["maps/10x10/basesWorkers10x10.xml"],
    reward_weight=np.array([10.0, 1.0, 1.0, 0.2, 1.0, 4.0]),
)
obs_micro_rts = micro_rts_env.reset()

def softmax(x, axis=None):
    x = x - x.max(axis=axis, keepdims=True)
    y = np.exp(x)
    return y / y.sum(axis=axis, keepdims=True)


def sample(logits):
    p = softmax(logits, axis=1)
    c = p.cumsum(axis=1)
    u = np.random.rand(len(c), 1)
    choices = (u < c).argmax(axis=1)
    return choices.reshape(-1, 1)

ai0 = RushAI(0,"Light", 10, 10)
ai1 = RushAI(1,"Light", 10, 10)

obs_nano_rts = nano_rts_env.reset()

for step in range(1000):
    time.sleep(0.01)
    nano_rts_env.render()
    micro_rts_env.render()
    game = nano_rts_env.games[0]
    # action: Action(unit_pos:int, action_type:str, target_pos:int, produced_unit_type:UnitType=None)
    action0 = ai0.get_action(game)
    action1 = ai1.get_action(game)
    #action_lists: List[List[Action]]
    obs_nano_rts, _, _, _ = nano_rts_env.one_game_step(action0, action1)

    action0_micro_rts = action0.action_to_array(10, 10)
    action1_micro_rts = action1.action_to_array(10, 10)
    action_micro_rts = np.concatenate([action0_micro_rts, action1_micro_rts], axis=0).reshape(2, 100, 7)

    action_mask = micro_rts_env.get_action_mask()
    next_obs, reward, done, info = micro_rts_env.step(action_micro_rts)
    if action0.action_type is not None:
        print(str(action0))
    if action1.action_type is not None:
        print(str(action1))