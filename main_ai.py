from base import play_cpu_by_cpu, RandomAI
from train import DQNOthelloAI
import tensorflow as tf

loaded_model = tf.keras.models.load_model("othello_model2.keras")
dqn = DQNOthelloAI()
dqn.model = loaded_model
dqn.epsilon = 0  # 評価時は探索をオフに

# 人間との対戦
winners = []
for _ in range(0, 100):
    winners += [play_cpu_by_cpu(dqn, RandomAI(), 0)]

print(winners.count("B"))
