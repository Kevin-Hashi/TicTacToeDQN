from gym.envs.registration import register

register(
    id='tictactoe-v0',
    entry_point='tictactoe_env.envs:TicTacToe'
)
