from nanorts.game_env import GameEnv
from ais.nano_rts_ai import RushAI, RandomAI, RoleAI
from nanorts.render import Render
from mcts.mc_ai import MCAI
import time

if __name__ == "__main__":
    rewards_wrights = {'win': 10, 'harvest': 1, 'return': 1, 'produce': 1, 'attack': 1}
    num_envs = 1
    map_paths = ['maps\\8x8\\basesWorkers8x8.xml' for _ in range(num_envs)]
    max_steps=5000
    env = GameEnv(map_paths, rewards_wrights, max_steps)
    
    width = 8
    height = 8

    ai0 = RushAI(0, "Light", width, height)
    ai1 = RoleAI(1, "Light", width, height)

    r = Render(width, height)

    for _ in range(100000):
        time.sleep(0.01)
        r.draw(env.games[0])
        game = env.games[0]
        # action: Action(unit_pos:int, action_type:str, target_pos:int, produced_unit_type:UnitType=None)
        action0 = ai0.get_action(game)
        action1 = ai1.get_action(game)
        #action_lists: List[List[Action]]
        states, rewards, dones, winners = env.one_game_step(action0, action1)