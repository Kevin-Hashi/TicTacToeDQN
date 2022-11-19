import argparse
import glob
import json
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import sys
import timeit
import warnings

import gym
import numpy as np
from keras.api.keras import Sequential
from keras.api.keras.layers import (Activation, BatchNormalization, Dense,
                                    Dropout, Flatten)
from keras.api.keras.optimizers import Adam
from keras.callbacks import TensorBoard
from rl.agents.dqn import DQNAgent
from rl.callbacks import TrainEpisodeLogger
from rl.memory import SequentialMemory
from rl.policy import BoltzmannQPolicy

from tictactoe_env.envs import tictactoe_enviroment



parser = argparse.ArgumentParser()
parser.add_argument('--jsonpath', type=str, default='')
args = parser.parse_args()


class Logger(TrainEpisodeLogger):
    def on_episode_end(self, episode, logs):
        """ Compute and print training statistics of the episode when done """
        duration = timeit.default_timer() - self.episode_start[episode]
        episode_steps = len(self.observations[episode])

        # Format all metrics.
        metrics = np.array(self.metrics[episode])
        metrics_template = ''
        metrics_variables = []
        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            for idx, name in enumerate(self.metrics_names):
                if idx > 0:
                    metrics_template += ', '
                try:
                    value = np.nanmean(metrics[:, idx])
                    metrics_template += '{}: {:f}'
                except Warning:
                    value = '--'
                    metrics_template += '{}: {}'
                metrics_variables += [name, value]
        metrics_text = metrics_template.format(*metrics_variables)

        nb_step_digits = str(
            int(np.ceil(np.log10(self.params['nb_steps']))) + 1)
        template = '{step: ' + nb_step_digits + \
            'd}/{nb_steps}: episode: {episode}, duration: {duration:.3f}s, episode steps: {episode_steps:3}, steps per second: {sps:3.0f}, episode reward: {episode_reward:6.3f}, mean reward: {reward_mean:6.3f} [{reward_min:6.3f}, {reward_max:6.3f}], mean action: {action_mean:.3f} [{action_min:.3f}, {action_max:.3f}],  {metrics}'
        variables = {
            'step': self.step,
            'nb_steps': self.params['nb_steps'],
            'episode': episode + 1,
            'duration': duration,
            'episode_steps': episode_steps,
            'sps': float(episode_steps) / duration,
            'episode_reward': np.sum(self.rewards[episode]),
            'reward_mean': np.mean(self.rewards[episode]),
            'reward_min': np.min(self.rewards[episode]),
            'reward_max': np.max(self.rewards[episode]),
            'action_mean': np.mean(self.actions[episode]),
            'action_min': np.min(self.actions[episode]),
            'action_max': np.max(self.actions[episode]),
            'metrics': metrics_text
        }
        if episode % 200 == 0:
            print()
            print(template.format(**variables), end='')
        else:
            print('\r'+template.format(**variables), end='')

        # Free up resources.
        del self.episode_start[episode]
        del self.observations[episode]
        del self.rewards[episode]
        del self.actions[episode]
        del self.metrics[episode]
    def on_train_end(self,logs):
        print()
        super().on_train_end(logs)


def create_model(input_shape, output_shape):
    model = Sequential()
    model.add(Flatten(input_shape=(1,)+input_shape))
    model.add(Dense(16))
    model.add(Activation('relu'))
    model.add(Dense(16))
    model.add(Activation('relu'))
    model.add(Dropout(0.1))
    model.add(BatchNormalization())
    model.add(Dense(8))
    model.add(Activation('relu'))
    model.add(Dense(8))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dense(output_shape))
    model.add(Activation('linear'))
    return model


def get_weights_path(weights_name):
    path = None
    for f in glob.glob(weights_name):
        if f != "":
            path = f
            break
    return path


def make_json(path, **kargs):
    with open(path, 'w') as f:
        json.dump(kargs, f, indent=2, sort_keys=True, ensure_ascii=False)


def load_json(path, key_list):
    raw_dict = {}
    with open(path, 'r') as f:
        raw_dict = json.load(f)
    select_dict = {}
    for key in key_list:
        try:
            select_dict[key] = raw_dict[key]
        except KeyError as e:
            print(e, file=sys.stderr)
    return select_dict


def main():
    last_weights_name=""
    #last_weights_name = "tictactoe_dqn"
    new_weights_name = "tictactoe_dqn2"
    ENV_NAME = 'tictactoe-v0'
    loop_max = 10
    step_amount = 20000
    save_option_json_path = new_weights_name+'_option.json'
    if args.jsonpath != '':
        keylist = ['last_weights_name', 'new_weights_name',
                   'ENV_NAME', 'loop_max', 'step_amount', 'save_json_path']
        json_dict = load_json(args.jsonpath, keylist)
        last_weights_name = json_dict['last_weights_name']
        new_weights_name = json_dict['new_weights_name']
        ENV_NAME = json_dict['ENV_NAME']
        loop_max = json_dict['loop_max']
        step_amount = json_dict['step_amount']
        save_option_json_path = json_dict['save_option_json_path']

    env = gym.make(ENV_NAME)
    model = create_model(env.get_obs_space().shape, env.get_action_space().n)
    memory = SequentialMemory(limit=5000000, window_length=1)
    policy = BoltzmannQPolicy()

    dqn_1 = DQNAgent(model=model, memory=memory, policy=policy,
                     nb_actions=env.get_action_space().n, nb_steps_warmup=32, gamma=0.3)
    dqn_1.compile(Adam(learning_rate=1e-3), metrics=['accuracy', 'mae'])

    os.makedirs(os.path.join('log', new_weights_name), exist_ok=True)
    model_postfix_number = int(sorted([f for f in os.listdir(os.path.join('log', new_weights_name)) if os.path.isdir(os.path.join(
        'log', new_weights_name, f))], key=lambda x: int(x))[-1])+1 if os.listdir(os.path.join('log', new_weights_name)) else 0
    for i in range(1, loop_max+1):
        tensorborad = TensorBoard(log_dir=os.path.join(
            'log', new_weights_name, str(model_postfix_number), str(i)))
        tensorborad._should_trace = True
        logger = Logger()
        print(f"loop {i}/{loop_max}")
        env.reset_all()
        w_path = get_weights_path(last_weights_name+'.h5')
        if w_path != None:
            dqn_1.load_weights(w_path)
        dqn_1.fit(env, nb_steps=step_amount, visualize=True,
                  verbose=0, callbacks=[tensorborad, logger])
        dqn_1.save_weights(new_weights_name+'.h5', overwrite=True)
        dqn_1.model.save(new_weights_name+'_model.h5', overwrite=True)
    make_json(save_option_json_path, last_weights_name=last_weights_name, new_weights_name=new_weights_name, ENV_NAME=ENV_NAME, loop_max=loop_max,
              step_amount=step_amount, save_option_json_path=save_option_json_path, model_postfix_number=model_postfix_number)


if __name__ == '__main__':
    main()
