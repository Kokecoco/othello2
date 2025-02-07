import tensorflow as tf
import numpy as np
from tqdm import tqdm
from base import *
from train import DQNOthelloAI
from collections import deque
import os
import sys
from minimax2 import Minimax2AI

# 警告抑制とGPU設定
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def setup_gpu():
    """GPUの初期化と最適化設定"""
    try:
        physical_devices = tf.config.list_physical_devices("GPU")
        if physical_devices:
            for device in physical_devices:
                tf.config.experimental.set_memory_growth(device, True)
            print(f"GPU設定完了: {physical_devices}")
            # RTX 4060向けの混合精度設定
            tf.keras.mixed_precision.set_global_policy("mixed_float16")
            return True
    except Exception as e:
        print(f"GPU設定エラー: {e}")
        return False


class OthelloEnv:
    """単一環境のラッパークラス"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.game = Board()
        self.turn = Board.B
        self.done = False
        # 形状を(8, 8, 1)に統一
        return np.array(self.game.board, dtype=np.float32).reshape(8, 8, 1) * self.turn

    def get_valid_moves(self):
        return self.game.get_valid_moves(self.turn)

    def step(self, action):
        if action is None:
            self.turn *= -1
            return (
                np.array(self.game.board, dtype=np.float32).reshape(8, 8, 1)
                * self.turn,
                0,
                self.done,
            )

        self.game.set_stone_on_board(action[0] - 1, action[1] - 1, self.turn)
        next_state = (
            np.array(self.game.board, dtype=np.float32).reshape(8, 8, 1) * self.turn
        )
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
    """並列環境管理クラス"""

    def __init__(self, num_envs):
        self.num_envs = num_envs
        self.envs = [OthelloEnv() for _ in range(num_envs)]

    @tf.function
    def reset(self):
        states = [env.reset() for env in self.envs]
        states_array = np.array(states, dtype=np.float32)
        return tf.convert_to_tensor(states_array, dtype=tf.float32)

    @tf.function
    def step(self, actions):
        results = [env.step(action) for env, action in zip(self.envs, actions)]
        states, rewards, dones = zip(*results)
        states_array = np.array(states, dtype=np.float32)
        rewards_array = np.array(rewards, dtype=np.float32)
        dones_array = np.array(dones, dtype=bool)
        return (
            tf.convert_to_tensor(states_array, dtype=tf.float32),
            tf.convert_to_tensor(rewards_array, dtype=tf.float32),
            tf.convert_to_tensor(dones_array, dtype=tf.bool),
        )


class ParallelDQNOthelloAI(DQNOthelloAI):
    def __init__(
        self,
        num_envs=6,
        learning_rate=0.001,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995,
        memory_size=10000,
        batch_size=32,
        target_update_freq=1000,
    ):
        self.num_envs = num_envs
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.memory = deque(maxlen=memory_size)
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.step_counter = 0

        # GPU最適化設定
        self._setup_gpu()

        # モデル構築
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_network()

    def _setup_gpu(self):
        """GPU設定の最適化"""
        try:
            physical_devices = tf.config.list_physical_devices("GPU")
            if physical_devices:
                tf.config.experimental.set_memory_growth(physical_devices[0], True)
                tf.keras.mixed_precision.set_global_policy("mixed_float16")
        except Exception as e:
            print(f"GPU設定エラー: {e}")

    def get_moves_batch(self, valid_moves_list, states):
        """バッチ処理による行動選択の最適化"""
        try:
            if tf.is_tensor(states):
                states = states.numpy()

            actions = []
            q_values = self.predict_batch(states)

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
        except Exception as e:
            print(f"行動選択エラー: {e}")
            return [None] * len(valid_moves_list)

    @tf.function
    def predict_batch(self, states):
        """最適化されたバッチ予測"""
        return self.model(states, training=False)

    def monitor_performance(self):
        """パフォーマンスモニタリング"""
        try:
            memory_info = tf.config.experimental.get_memory_info("GPU:0")
            memory_usage = memory_info["current"] / 1024**2
            return memory_usage
        except:
            return None


def debug_step_values(state, reward, done):
    """ステップ実行結果の検証"""
    print(f"State shape: {state.shape}, dtype: {state.dtype}")
    print(f"Reward: {reward}, type: {type(reward)}")
    print(f"Done: {done}, type: {type(done)}")


def monitor_resources():
    """システムリソースのモニタリング"""
    try:
        memory_info = tf.config.experimental.get_memory_info("GPU:0")
        print(f"GPU使用メモリ: {memory_info['current'] / 1024**2:.2f}MB")
        return (
            memory_info["current"] / memory_info["peak"]
            if memory_info["peak"] > 0
            else 0
        )
    except:
        return None


def train_ai_parallel(episodes=100000, num_envs=6):
    """RTX 4060向けに最適化された学習処理"""
    if not setup_gpu():
        print("警告: GPU設定失敗")
        return None

    def adjust_batch_size(current_batch_size, memory_usage):
        if memory_usage > 0.9:
            return max(16, current_batch_size // 2)
        elif memory_usage < 0.5:
            return min(128, current_batch_size * 2)
        return current_batch_size

    try:
        ai = ParallelDQNOthelloAI(num_envs=num_envs)
        env = ParallelOthelloEnv(num_envs)
        minimax_ai = Minimax2AI(4)
        current_batch_size = 32

        with tf.device("/GPU:0"):
            for episode in tqdm(range(0, episodes, num_envs)):
                if episode % 100 == 0:
                    memory_usage = monitor_resources()
                    if memory_usage is not None:
                        current_batch_size = adjust_batch_size(
                            current_batch_size, memory_usage
                        )
                        ai.batch_size = current_batch_size
                        print(f"バッチサイズ調整: {current_batch_size}")

                try:
                    states = env.reset()
                    episode_done = [False] * num_envs
                    turns = [Board.B] * num_envs

                    while not all(episode_done):
                        actions = []
                        for i in range(num_envs):
                            if turns[i] == Board.B:
                                valid_moves = env.envs[i].get_valid_moves()
                                action = ai.get_move(valid_moves, states[i], Board.B)
                            else:
                                valid_moves = env.envs[i].get_valid_moves()
                                action = minimax_ai.get_move(
                                    valid_moves, states[i], Board.W
                                )
                            actions.append(action)
                        next_states, rewards, dones = env.step(actions)
                        turns = np.array(turns)

                        for i, (state, action, reward, next_state, done) in enumerate(
                            zip(
                                states.numpy(),
                                actions,
                                rewards.numpy(),
                                next_states.numpy(),
                                dones.numpy(),
                            )
                        ):
                            if turns[i] == Board.B and action is not None:
                                ai.remember(state, action, reward, next_state, done)
                                ai.replay()

                            if not done:
                                turns[i] *= -1  # ターンを切り替え

                        states = next_states
                        episode_done = [
                            done or ed for done, ed in zip(dones.numpy(), episode_done)
                        ]

                except tf.errors.ResourceExhaustedError:
                    print("GPUメモリ不足。バッチサイズを削減します。")
                    current_batch_size = max(16, current_batch_size // 2)
                    ai.batch_size = current_batch_size
                    continue
                except Exception as e:
                    print(e)
                    raise

                if (episode + num_envs) % 100 == 0:
                    print(
                        f"Episode: {episode + num_envs}/{episodes}, "
                        f"Epsilon: {ai.epsilon:.2f}"
                    )

        ai.model.save(f"othello_model_{episodes}.keras")
        return ai

    except Exception as e:
        print(f"エラーが発生しました: {e}")
        raise
        return None


if __name__ == "__main__":
    print("GPU Available:", tf.config.list_physical_devices("GPU"))
    model = train_ai_parallel(episodes=int(sys.argv[1]), num_envs=int(sys.argv[2]))
