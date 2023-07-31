from nanorts.game_env import GameEnv
from ais.nano_rts_ai import RushAI, RandomAI
from nanorts.render import Render
from mcts.mc_ai import MCAI

if __name__ == "__main__":
    rewards_wrights = {'win': 10, 'harvest': 1, 'return': 1, 'produce': 1, 'attack': 1}
    num_envs = 1
    map_paths = ['maps\\8x8\\basesWorkers8x8.xml' for _ in range(num_envs)]
    max_steps=5000
    env = GameEnv(map_paths, rewards_wrights, max_steps)
    
    width = 8
    height = 8

    playou_ai1 = RandomAI(0)
    playou_ai2 = RandomAI(1)

    ai0 = MCAI(playou_ai1, playou_ai2, 0, num_playout = 20)

    ai1 = RandomAI(1)

    r = Render(16,16)

    for _ in range(100000):
        r.draw(env.games[0])
        game = env.games[0]
        # action: Action(unit_pos:int, action_type:str, target_pos:int, produced_unit_type:UnitType=None)
        action0 = ai0.get_action(game)
        action1 = ai1.get_action(game)
        #action_lists: List[List[Action]]
        states, rewards, dones, winners = env.one_game_step(action0, action1)