from base import MaxAI, CornerAI, play_cpu_by_cpu
from minimax import MinimaxAI
from mcts import MonteCarloAI

m = MaxAI()
c = CornerAI()
mm2 = MinimaxAI(2)
mm3 = MinimaxAI(3)
mm4 = MinimaxAI(4)
mcts2 = MonteCarloAI(2)
mcts3 = MonteCarloAI(3)
mcts4 = MonteCarloAI(4)
cpus = [m, c, mm2, mm3, mm4, mcts2, mcts3, mcts4]


def get_data():
    global results
    for i in range(len(cpus)):
        for j in range(len(cpus)):
            wins = [0, 0, 0]
            for _ in range(100):
                winner = play_cpu_by_cpu(cpus[i], cpus[j], False)
                if winner == "B":
                    wins[0] += 1
                elif winner == "W":
                    wins[1] += 1
                else:
                    wins[2] += 1
            cpu1name = cpus[i].__class__.__name__
            cpu1name = (
                cpu1name
                if (cpu1name != "MinimaxAI" and cpu1name != "MonteCarloAI")
                else cpu1name + "(" + str(cpus[i].evaluater) + ")"
            )
            cpu2name = cpus[j].__class__.__name__
            cpu2name = (
                cpu2name
                if (cpu2name != "MinimaxAI" and cpu2name != "MonteCarloAI")
                else cpu2name + "(" + str(cpus[j].evaluater) + ")"
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
