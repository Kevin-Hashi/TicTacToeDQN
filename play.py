import numpy as np
from keras.api.keras.models import load_model
from tictactoe_env.envs import TicTacToe
from game import create_model
import gym
from keras.api.keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory


def input_place():
    place = input("選択する場所 0:LT,1:MT,2:RT,3:LM,4:MM,5:RM,6:LB,7,MB,8:RB : ")
    assert int(place) in {0, 1, 2, 3, 4, 5, 6, 7,
                          8}, "place must be in range between 0 and 8."
    return int(place)


def print_game(state):
    print(' '.join([str(int(n)) for n in state[:3]]))
    print(' '.join([str(int(n)) for n in state[3:6]]))
    print(' '.join([str(int(n)) for n in state[6:]]))


def main():
    weights_path = "tictactoe_dqn2.h5"
    ENV_NAME = 'tictactoe-v0'
    env = gym.make(ENV_NAME)
    #model = create_model(env.get_obs_space().shape, env.get_action_space().n)
    #model.load_weights(weights_path)
    model=load_model(weights_path+'_model.h5')
    memory = SequentialMemory(limit=5000000, window_length=1)
    policy = BoltzmannQPolicy()

    dqn_1 = DQNAgent(model=model, memory=memory, policy=policy,
                     nb_actions=env.get_action_space().n, nb_steps_warmup=32, gamma=0.3)
    dqn_1.compile(Adam(lr=1e-3), metrics=['accuracy', 'mae'])
    game = TicTacToe()
    while True:
        try:
            game.reset_all()
            player_number = int(input("先攻:1, 後攻:2, 終了:3 : "))
            assert player_number in {
                1, 2, 3}, "Player number must be 1, 2 or 3."
            if player_number == 3:
                break
            while not game.board_is_full():
                result = None
                if game.actual_player == player_number:
                    place = 100
                    while int(place) not in {0, 1, 2, 3, 4, 5, 6, 7, 8} or game.state[place] != 0:
                        place = input_place()
                    result = game.step(place)
                else:
                    place = 100
                    while int(place) not in {0, 1, 2, 3, 4, 5, 6, 7, 8} or game.state[place] != 0:
                        place = dqn_1.forward(game.state)
                    result = game.step(place)
                print_game(game.state)
                print()
                if result[2] == True:
                    break
        except KeyboardInterrupt:
            break
        else:
            print("End of game")


if __name__ == "__main__":
    main()
