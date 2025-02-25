from base import play_cpu_by_cpu
from train import DQNOthelloAI
import tensorflow as tf
from tqdm import tqdm


def power_of_ten(limit):
    num = 1
    for _ in range(limit):
        yield num
        num *= 10


cpus = []
for i in power_of_ten(7):
    cpus += [DQNOthelloAI()]
    cpus[-1].model = tf.keras.models.load_model(f"othello_model_{i}.keras")

loaded_model = tf.keras.models.load_model("othello_model2.keras")
dqn = DQNOthelloAI()
dqn.model = loaded_model
dqn.epsilon = 0  # 評価時は探索をオフに


def get_data():
    global results
    for i in range(len(cpus)):
        for j in range(len(cpus)):
            wins = [0, 0, 0]
            for _ in tqdm(range(1000)):
                winner = play_cpu_by_cpu(cpus[i], cpus[j], False)
                if winner == "B":
                    wins[0] += 1
                elif winner == "W":
                    wins[1] += 1
                else:
                    wins[2] += 1
            cpu1name = cpus[i].__class__.__name__ + str(10**i)
            cpu2name = cpus[j].__class__.__name__ + str(10**j)
            print(
                f"{cpu1name} by {cpu2name} ",
                "B:",
                wins[0],
                ", W:",
                wins[1],
                ", Draw:",
                wins[2],
                sep="",
            )
            results += [
                f"{cpu1name} by {cpu2name} B:",
                f"{wins[0]} W:{wins[1]} Draw:{wins[2]}",
            ]
        print()
        results += []


results = []
get_data()
print(results)
