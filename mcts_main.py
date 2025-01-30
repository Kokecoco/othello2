"""
モンテカルロ木探索で対戦させる
"""
from base import play_cpu_by_cpu
from mcts import MonteCarloAI

mcts2 = MonteCarloAI(2, 1000)
mcts3 = MonteCarloAI(3, 1000)
mcts4 = MonteCarloAI(4, 1000)
cpus = [mcts2, mcts3, mcts4]


def get_data():
    """
    get_data
    """
    global results
    for i in range(len(cpus)):
        for j in range(len(cpus)):
            wins = [0, 0, 0]
            for _ in range(10):
                winner = play_cpu_by_cpu(cpus[i], cpus[j], False)
                if winner == "B":
                    wins[0] += 1
                elif winner == "W":
                    wins[1] += 1
                else:
                    wins[2] += 1
            cpu1name = cpus[i].__class__.__name__
            cpu1name = (
                cpu1name + "(" + str(cpus[i].evaluater) + ")"
            )
            cpu2name = cpus[j].__class__.__name__
            cpu2name = (
                cpu2name + "(" + str(cpus[j].evaluater) + ")"
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
                f"{cpu1name} by {cpu2name} B:{wins[0]} W:{wins[1]} Draw:{wins[2]}"
            ]
        print()
        results += []


results = []
get_data()
