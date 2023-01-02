from gymnasium.envs.registration import register


register(
    id='TicTacToe-3x3-v0',
    entry_point='tictactoe.envs:TicTacToeEnv3x3',
    max_episode_steps=100,
)