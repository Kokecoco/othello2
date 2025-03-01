from base import MaxAI, CornerAI, play_cpu_by_cpu
from minimax import MinimaxAI
from train import DQNOthelloAI
from mcts import MonteCarloAI
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

m = MaxAI()
c = CornerAI()
mm2 = MinimaxAI(2)
mm3 = MinimaxAI(3)
mm4 = MinimaxAI(4)
mcts = MonteCarloAI(2)
cpus += [m, c, mm2, mm3, mm4, mcts]


def get_data():
    global results
    for i in range(len(cpus)):
        for j in range(len(cpus)):
            wins = [0, 0, 0]
            num_battle = (
                10
                if cpus[i].__class__.__name__ == "MonteCarloAI"
                or cpus[j].__class__.__name__ == "MonteCarloAI"
                else 100
            )
            for _ in tqdm(range(num_battle)):
                winner = play_cpu_by_cpu(cpus[i], cpus[j], False)
                if winner == "B":
                    wins[0] += 1
                elif winner == "W":
                    wins[1] += 1
                else:
                    wins[2] += 1
            cpu1name = cpus[i].__class__.__name__
            cpu1name = (
                (
                    cpu1name
                    if (cpu1name != "MinimaxAI" and cpu1name != "MonteCarloAI")
                    else cpu1name + "(" + str(cpus[i].evaluater) + ")"
                )
                if cpu1name != "DQNOthelloAI"
                else cpu1name + str(10**i)
            )
            cpu2name = cpus[j].__class__.__name__
            cpu2name = (
                (
                    cpu2name
                    if (cpu2name != "MinimaxAI" and cpu2name != "MonteCarloAI")
                    else cpu2name + "(" + str(cpus[j].evaluater) + ")"
                )
                if cpu2name != "DQNOthelloAI"
                else cpu2name + str(10**j)
            )
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
