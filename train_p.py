import tensorflow as tf
import numpy as np
from collections import deque
import random
from tqdm import tqdm
from base import *
from train import DQNOthelloAI


class OthelloEnv:
    """ベクトル化可能なオセロ環境"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.game = Board()
        self.turn = Board.B
        self.done = False
        return np.array(self.game.board) * self.turn

    def step(self, action):
        if action is None:
            self.turn *= -1
            return np.array(self.game.board) * self.turn, 0, self.done

        self.game.set_stone_on_board(action[0] - 1, action[1] - 1, self.turn)
        next_state = np.array(self.game.board) * self.turn

        reward = self._calculate_reward(action)
        self._check_done()

        self.turn *= -1
        return next_state, reward, self.done

    def _calculate_reward(self, action):
        b_count = sum(row.count(Board.B) for row in self.game.board)
        w_count = sum(row.count(Board.W) for row in self.game.board)
        stone_diff = (
            (b_count - w_count) if self.turn == Board.B else (w_count - b_count)
        )

        reward = stone_diff * 0.1

        if (action[0] - 1, action[1] - 1) in [(0, 0), (0, 7), (7, 0), (7, 7)]:
            reward += 5

        if self.done and stone_diff > 0:
            reward += 10
        elif self.done and stone_diff < 0:
            reward -= 10
        return reward

    def _check_done(self):
        if not self.game.get_valid_moves(
            self.turn * -1
        ) and not self.game.get_valid_moves(self.turn):
            self.done = True


class ParallelOthelloEnv:
    """複数の環境を並列実行するためのラッパー"""

    def __init__(self, num_envs):
        self.num_envs = num_envs
        self.envs = [OthelloEnv() for _ in range(num_envs)]

    def reset(self):
        """すべての環境をリセットし、初期状態を返す"""
        states = [env.reset() for env in self.envs]
        return tf.convert_to_tensor(states, dtype=tf.float32)

    @tf.function
    def step(self, actions):
        """バッチ処理による並列ステップ実行"""
        results = [env.step(action) for env, action in zip(self.envs, actions)]
        states, rewards, dones = zip(*results)
        return (
            tf.convert_to_tensor(states, dtype=tf.float32),
            tf.convert_to_tensor(rewards, dtype=tf.float32),
            tf.convert_to_tensor(dones, dtype=tf.bool),
        )


class ParallelDQNOthelloAI(DQNOthelloAI):
    """並列処理に対応したDQN実装"""

    def __init__(self, num_envs=8, **kwargs):
        super().__init__(**kwargs)
        self.num_envs = num_envs

    @tf.function
    def predict_batch(self, states):
        """バッチ予測の最適化"""
        return self.model(states, training=False)

    def get_moves_batch(self, valid_moves_list, states):
        """バッチ処理による行動選択"""
        q_values = self.predict_batch(states)
        actions = []

        for i, moves in enumerate(valid_moves_list):
            if not moves:
                actions.append(None)
                continue

            if random.random() <= self.epsilon:
                actions.append(random.choice(moves))
            else:
                valid_moves_q = [
                    (move, q_values[i][(move[0] - 1) * 8 + (move[1] - 1)].numpy())
                    for move in moves
                ]
                actions.append(max(valid_moves_q, key=lambda x: x[1])[0])

        return actions


def train_ai_parallel(episodes=5000, num_envs=8):
    """並列処理による学習の実行"""
    # GPUの利用可能性を確認
    physical_devices = tf.config.list_physical_devices("GPU")
    if physical_devices:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    ai = ParallelDQNOthelloAI(num_envs=num_envs)
    env = ParallelOthelloEnv(num_envs)

    with tf.device("/GPU:0"):
        for episode in tqdm(range(0, episodes, num_envs)):
            states = env.reset()
            episode_done = [False] * num_envs

            while not all(episode_done):
                try:
                    valid_moves_list = [env.get_valid_moves() for env in env.envs]
                    actions = ai.get_moves_batch(valid_moves_list, states)

                    next_states, rewards, dones = env.step(actions)

                    # 経験の保存とリプレイ
                    for state, action, reward, next_state, done in zip(
                        states, actions, rewards, next_states, dones
                    ):
                        if action is not None:
                            ai.remember(state, action, reward, next_state, done)
                            ai.replay()

                    states = next_states
                    episode_done = [done or ed for done, ed in zip(dones, episode_done)]

                except Exception as e:
                    print(f"Error during training: {e}")
                    continue

            if (episode + num_envs) % 100 == 0:
                print(
                    f"Episode: {episode + num_envs}/{episodes}, "
                    f"Epsilon: {ai.epsilon:.2f}"
                )

    ai.model.save("othello_model.keras")
    return ai


if __name__ == "__main__":
    # GPUの確認
    print("GPU Available:", tf.config.list_physical_devices("GPU"))

    # 学習の実行（エピソード数と並列環境数を指定）
    model = train_ai_parallel(episodes=5000, num_envs=6)
