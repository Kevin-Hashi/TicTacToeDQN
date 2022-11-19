import gym
import gym.spaces
import numpy as np
import datetime


class TicTacToe(gym.core.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.actual_player = 1
        self.total_steps = 0
        self.valid_move = 0
        self.invalid_move = 0
        self.tris_amount = {}
        self.game = 0

        self.state = np.zeros(9)
        self.action_space = gym.spaces.Discrete(9)
        self.obs_min = np.full((9,), 0)
        self.obs_max = np.full((9,), 2)
        self.observation_space = gym.spaces.Box(
            low=self.obs_min, high=self.obs_max, dtype=np.int8)

        self.debug_file_name = "TicTacToe_debug.txt"
        self.report_file_name = "TicTacToe_report.txt"
        self.debug = open(self.debug_file_name, "a+")
        self.report = open(self.report_file_name, "w+")

    def step(self, action):
        self.debug.write(
            "\n----------------------------------------------------------------\n")
        self.debug.write('\n'+str(datetime.datetime.now().time())+"\n")
        self.debug.write('\nActual Player'+str(self.actual_player)+'\n')
        self.debug.write('\nCurrent State'+str(self.state)+'\n')
        done = False
        reward = 0
        self.increment_steps()

        if not self.board_is_full():
            if (self.state[action] == 0):
                state = self.state
                state[action] = self.actual_player
                self.state = state
                reward = 1
                self.debug.write("\nGood Move.\n")
                self.debug.write("\nNew State "+str(self.state) +
                                 " Reward "+str(reward)+"\n")
                self.valid_move += 1

                tris, player, pos = self.tris()
                if tris and player == self.actual_player:
                    reward += 29#+sum(1 for x in self.state if x == 0)
                    self.increment_tris_amount(player)
                    self.debug.write("\nTris = "+str(tris)+" player "+str(player) +
                                     " pos "+str(pos)+" reward total:"+str(reward)+"\n")
                    done = True
                block, player, pos = self.block()
                if block and player == self.actual_player:
                    reward += block
                    self.debug.write(
                        f'\nBlock = {block} player {player} pos {pos} reward total:{reward}')
                self.change_player()
            else:
                self.increment_invalid_move()
                reward -= 10
                self.debug.write("\nInvalid move. Reward:"+str(reward)+" Invalid move "+str(self.valid_move) +
                                 " Total Step "+str(self.total_steps)+" ratio "+str(self.invalid_move/self.total_steps)+"\n")
        else:
            self.debug.write("\nFull Game Board.\n")
            self.debug.write("\nDraw Game:"+str(self.state)+"\n")
            reward = 0
            done = True
        self.debug.write("\nNext Player: "+str(self.actual_player)+"\n")
        self.debug.write(
            "\n--------------------------------------------------------\n")
        self.debug.flush()
        return np.array(self.state), reward, done, {}

    def reset(self):
        state = np.zeros(9)
        self.state = state
        self.debug.write("\nEnvironment Reset\n")
        self.debug.write('\n'+str(datetime.datetime.now().time())+"\n")
        self.debug.flush()
        self.game += 1
        return np.array(self.state)

    def reset_all(self):
        self.debug.close()
        self.debug = open(self.debug_file_name, "w+")
        if self.game != 0:
            self.report.write("\n----------REPORT----------\n")
            self.report.write('\n'+str(datetime.datetime.now().time())+"\n")
            self.report.write("\nepisode played "+str(self.total_steps)+"\n")
            self.report.write("\nNumber of matches "+str(self.game)+"\n")
            self.report.write("Number of tris"+str(self.tris_amount)+"\n")
            self.report.write("Match/tris ratio" +
                              str(sum(self.tris_amount)/self.game)+"\n")
            self.report.write("Invalid move"+str(self.invalid_move)+"\n")
            self.report.write("Valid move"+str(self.valid_move)+"\n")
            self.report.write("Episode/Invalid move ratio" +
                              str(self.invalid_move/self.total_steps)+"\n")
            self.report.write("Episode/Valid move ratio" +
                              str(self.valid_move/self.total_steps)+"\n")
            self.report.write("\n----------END REPORT----------\n")
        self.report.flush()
        state = np.zeros(9)
        self.state = state
        self.actual_player = 1
        self.total_steps = 0
        self.invalid_move = 0
        self.tris_amount = {}
        self.game = 0
        self.valid_move = 0

    def get_obs_space(self):
        return self.observation_space

    def get_action_space(self):
        return self.action_space

    def increment_steps(self):
        self.total_steps += 1

    def increment_invalid_move(self):
        self.invalid_move += 1

    def increment_tris_amount(self, player):
        if (self.tris_amount.get(player) == None):
            self.tris_amount.update({player: 1})
        else:
            actual_val = self.tris_amount.get(player)
            new_val = actual_val+1
            self.tris_amount[player] = new_val

    def change_player(self):
        if self.actual_player == 1:
            self.actual_player = 2
        elif self.actual_player == 2:
            self.actual_player = 1
        else:
            pass

    def board_is_full(self):
        # return not np.any(self.state == 0)
        for val in self.state:
            if val == 0.0:
                return False
        return True

    def tris(self):
        tris = False
        player = 0
        pos = False

        tris, player, pos = self.tris_for_rows()
        if tris:
            return tris, player, pos

        tris, player, pos = self.tris_for_cols()
        if tris:
            return tris, player, pos

        tris, player, pos = self.tris_for_diags()
        if tris:
            return tris, player, pos

        return tris, player, pos

    def tris_for_rows(self):
        tris = False
        player = 0
        pos = None
        actual_player = self.actual_player
        if self.state[0] == actual_player and self.state[1] == actual_player and self.state[2] == actual_player:
            tris = True
            player = actual_player
            pos = "first row"
        elif self.state[3] == actual_player and self.state[4] == actual_player and self.state[5] == actual_player:
            tris = True
            player = actual_player
            pos = "second row"
        elif self.state[6] == actual_player and self.state[7] == actual_player and self.state[8] == actual_player:
            tris = True
            player = actual_player
            pos = "third row"
        else:
            pass

        return tris, player, pos

    def tris_for_cols(self):
        tris = False
        player = 0
        pos = None
        actual_player = self.actual_player
        if self.state[0] == actual_player and self.state[3] == actual_player and self.state[6] == actual_player:
            tris = True
            player = actual_player
            pos = "first col"
        elif self.state[1] == actual_player and self.state[4] == actual_player and self.state[7] == actual_player:
            tris = True
            player = actual_player
            pos = "second col"
        elif self.state[2] == actual_player and self.state[5] == actual_player and self.state[8] == actual_player:
            tris = True
            player = actual_player
            pos = "third col"
        else:
            pass

        return tris, player, pos

    def tris_for_diags(self):
        tris = False
        player = 0
        pos = None
        actual_player = self.actual_player
        if self.state[0] == actual_player and self.state[4] == actual_player and self.state[8] == actual_player:
            tris = True
            player = actual_player
            pos = "l to r diag"
        elif self.state[2] == actual_player and self.state[4] == actual_player and self.state[6] == actual_player:
            tris = True
            player = actual_player
            pos = "r to l diag"
        else:
            pass

        return tris, player, pos

    def block(self):
        block = 0
        player = 0
        pos = None
        block_count = 0
        pos_str = ""

        block, player, pos = self.block_for_rows()
        pos_str += pos
        block_count += block
        block, player, pos = self.block_for_cols()
        pos_str += pos
        block_count += block
        block, player, pos = self.block_for_diags()
        pos_str += pos
        block_count += block
        return block_count, player, pos_str

    def block_for_rows(self):
        block = 0
        player = 0
        pos = ''
        actual_player = self.actual_player
        if self.state[0] == actual_player and self.state[1] == actual_player and self.state[2] != actual_player and self.state[2] != 0\
                or self.state[0] == actual_player and self.state[1] != actual_player and self.state[1] != 0 and self.state[2] == actual_player\
                or self.state[0] != actual_player and self.state[0] != 0 and self.state[1] == actual_player and self.state[2] == actual_player:
            block += 1
            player = actual_player
            pos += "first row "
        if self.state[3] == actual_player and self.state[4] == actual_player and self.state[5] != actual_player and self.state[5] != 0\
                or self.state[3] == actual_player and self.state[4] != actual_player and self.state[4] != 0 and self.state[5] == actual_player\
                or self.state[3] != actual_player and self.state[3] != 0 and self.state[4] == actual_player and self.state[5] == actual_player:
            block += 1
            player = actual_player
            pos += "second row "
        if self.state[6] == actual_player and self.state[7] == actual_player and self.state[8] != actual_player and self.state[8] != 0\
                or self.state[6] == actual_player and self.state[7] != actual_player and self.state[7] != 0 and self.state[8] == actual_player\
                or self.state[6] != actual_player and self.state[6] != 0 and self.state[7] == actual_player and self.state[8] == actual_player:
            block += 1
            player = actual_player
            pos += "third row "

        return block, player, pos

    def block_for_cols(self):
        block = 0
        player = 0
        pos = ''
        actual_player = self.actual_player
        if self.state[0] == actual_player and self.state[3] == actual_player and self.state[6] != actual_player and self.state[6] != 0\
                or self.state[0] == actual_player and self.state[3] != actual_player and self.state[3] != 0 and self.state[6] == actual_player\
                or self.state[0] != actual_player and self.state[0] != 0 and self.state[3] == actual_player and self.state[6] == actual_player:
            block += 1
            player = actual_player
            pos += 'first col '
        if self.state[1] == actual_player and self.state[4] == actual_player and self.state[7] != actual_player and self.state[7] != 0\
                or self.state[1] == actual_player and self.state[4] != actual_player and self.state[4] != 0 and self.state[7] == actual_player\
                or self.state[1] != actual_player and self.state[1] != 0 and self.state[4] == actual_player and self.state[7] == actual_player:
            block += 1
            player = actual_player
            pos += 'second col '
        if self.state[2] == actual_player and self.state[5] == actual_player and self.state[8] != actual_player and self.state[8] != 0\
                or self.state[2] == actual_player and self.state[5] != actual_player and self.state[5] != 0 and self.state[8] == actual_player\
                or self.state[2] != actual_player and self.state[2] != 0 and self.state[5] == actual_player and self.state[8] == actual_player:
            block += 1
            player = actual_player
            pos += 'third col '

        return block, player, pos

    def block_for_diags(self):
        block = 0
        player = 0
        pos = ''
        actual_player = self.actual_player
        if self.state[0] == actual_player and self.state[4] == actual_player and self.state[8] != actual_player and self.state[8] != 0\
                or self.state[0] == actual_player and self.state[4] != actual_player and self.state[4] != 0 and self.state[8] == actual_player\
                or self.state[0] != actual_player and self.state[0] != 0 and self.state[4] == actual_player and self.state[8] == actual_player:
            block += 1
            player = actual_player
            pos += 'l to r diag '
        if self.state[2] == actual_player and self.state[4] == actual_player and self.state[6] != actual_player and self.state[6] != 0\
                or self.state[2] == actual_player and self.state[4] != actual_player and self.state[4] != 0 and self.state[6] == actual_player\
                or self.state[2] != actual_player and self.state[2] != 0 and self.state[4] == actual_player and self.state[6] == actual_player:
            block += 1
            player = actual_player
            pos += 'r to l diag '

        return block, player, pos

    def render(self, mode='human', close=False):
        pass

    def close(self):
        pass
