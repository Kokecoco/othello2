import re


def process_othello_results(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()

    results = []

    for line in lines:
        match = re.search(r"(.+) by (.+) B:(\d+) W:(\d+) Draw:(\d+)", line.strip())
        if match:
            ai1 = match.group(1)
            ai2 = match.group(2)
            b_wins = int(match.group(3))
            w_wins = int(match.group(4))

            # 勝利したAIを判定
            if b_wins > w_wins:
                winner = ai1
            elif w_wins > b_wins:
                winner = ai2
            else:
                winner = "Draw"  # 引き分けの場合

            results.append(f'"{ai1} by {ai2} {winner}"')

    return results


# テスト用（ファイル名を適宜変更）
file_path = "othello_results.txt"
output = process_othello_results(file_path)
for line in output:
    print(line)
