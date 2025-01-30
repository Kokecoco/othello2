from base import *

import tensorflow as tf
import numpy as np
from collections import deque
import random
from tqdm import tqdm


class DQNOthelloAI(AI):
    def __init__(
        self,
        learning_rate=0.001,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995,
        memory_size=10000,
        batch_size=32,
        target_update_freq=1000,
    ):
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.memory = deque(maxlen=memory_size)
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.step_counter = 0

        # メインネットワークとターゲットネットワークの構築
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_network()

    def _build_model(self):
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(8, 8, 1)),
                tf.keras.layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(256, activation="relu"),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Dense(64),
            ]
        )
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss="huber",
        )
        return model

    def update_target_network(self):
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def get_move(self, moves, board, stone):
        if not moves:
            return None

        state = np.array(board).reshape(1, 8, 8, 1) * stone

        if random.random() <= self.epsilon:
            return random.choice(moves)

        q_values = self.model.predict(state, verbose=0)[0]
        valid_moves_q = [
            (move, q_values[(move[0] - 1) * 8 + (move[1] - 1)]) for move in moves
        ]
        return max(valid_moves_q, key=lambda x: x[1])[0]

    def calculate_reward(self, board, stone, move, done):
        # 高度な報酬設計
        reward = 0

        # 基本的な石の数の差
        b_count = sum(row.count(Board.B) for row in board)
        w_count = sum(row.count(Board.W) for row in board)
        stone_diff = (b_count - w_count) if stone == Board.B else (w_count - b_count)
        reward += stone_diff * 0.1

        # 角の獲得に対する報酬
        corners = [(0, 0), (0, 7), (7, 0), (7, 7)]
        if (move[0] - 1, move[1] - 1) in corners:
            reward += 5

        # ゲーム終了時の報酬
        if done:
            if stone_diff > 0:
                reward += 10
            elif stone_diff < 0:
                reward -= 10

        return reward

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)
        states = np.zeros((self.batch_size, 8, 8, 1))
        next_states = np.zeros((self.batch_size, 8, 8, 1))

        for i, (state, _, _, next_state, _) in enumerate(minibatch):
            states[i] = state.reshape(8, 8, 1)
            next_states[i] = next_state.reshape(8, 8, 1)

        targets = self.model.predict(states, verbose=0)
        next_q_values = self.target_model.predict(next_states, verbose=0)

        for i, (_, action, reward, _, done) in enumerate(minibatch):
            if done:
                targets[i][(action[0] - 1) * 8 + (action[1] - 1)] = reward
            else:
                targets[i][(action[0] - 1) * 8 + (action[1] - 1)] = (
                    reward + self.gamma * np.max(next_q_values[i])
                )

        self.model.fit(states, targets, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        self.step_counter += 1
        if self.step_counter % self.target_update_freq == 0:
            self.update_target_network()


def train_ai(episodes=100):
    ai = DQNOthelloAI()

    for episode in tqdm(range(episodes)):
        game = Board()
        turn = Board.B
        done = False

        while not done:
            moves = game.get_valid_moves(turn)
            if not moves:
                if not game.get_valid_moves(-turn):
                    done = True
                turn *= -1
                continue

            current_state = np.array(game.board) * turn
            move = ai.get_move(moves, game.board, turn)

            # 手を打つ
            game.set_stone_on_board(move[0] - 1, move[1] - 1, turn)
            next_state = np.array(game.board) * turn

            # 報酬計算
            reward = ai.calculate_reward(game.board, turn, move, done)

            # 経験を記憶
            ai.remember(current_state, move, reward, next_state, done)

            # 学習
            ai.replay()

            turn *= -1

        if (episode + 1) % 100 == 0:
            print(
                f"""Episode: {
                    episode + 1}/{episodes}, Epsilon: {ai.epsilon:.2f}"""
            )

    # モデルの保存
    ai.model.save("othello_model.keras")
    return ai


if __name__ == "__main__":
    model = train_ai(5000)
