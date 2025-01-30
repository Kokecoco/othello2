import torch
import os
from base import Board
import random


def random_play(board, num_moves, player):
    """序盤はランダムに有効な手を選び、オセロとして有効な手のみを使用"""
    for _ in range(num_moves):
        valid_moves = board.get_valid_moves(player)
        if not valid_moves:
            player = -player
            valid_moves = board.get_valid_moves(player)
            if not valid_moves:
                break  # 両方とも置けないなら終了
        move = random.choice(valid_moves)
        board.set_stone_on_board(*move, player)
        player = -player  # プレイヤー交代
    return board, player


def calculate_win_rate(board, player):
    """勝率を計算する評価関数"""
    black_score = sum(1 for row in board.board for cell in row if cell == -1)
    white_score = sum(1 for row in board.board for cell in row if cell == 1)
    if black_score == white_score:
        return 0.5
    elif player == -1:
        return 1.0 if black_score > white_score else 0.0
    elif player == 1:
        return 1.0 if white_score > black_score else 0.0


def generate_tree_data(board, player, max_depth=3):
    """木探索によるデータ生成。探索はオセロとして有効な手のみを使用"""
    data = []
    stack = [(board, player, 0)]  # 深さをスタックに追加

    while stack:
        current_board, current_player, depth = stack.pop()
        valid_moves = current_board.get_valid_moves(current_player)

        if not valid_moves or depth >= max_depth:
            win_rate = calculate_win_rate(current_board, current_player)
            data.append(
                (torch.tensor(current_board.board, dtype=torch.float32), win_rate))
            continue

        for move in valid_moves:
            new_board = Board()
            new_board.board = [row[:] for row in current_board.board]
            new_board.set_stone_on_board(*move, current_player)
            stack.append((new_board, -current_player, depth + 1))
    return data


def save_dataset(num_games, random_moves, data_path="othello_data.pt", batch_size=10):
    """データセットを生成し、torch形式で保存する"""
    board_data = []
    score_data = []
    temp_files = []

    for game_idx in range(num_games):
        board = Board()
        player = random.choice([-1, 1])
        board, player = random_play(board, random_moves, player)
        tree_data = generate_tree_data(board, player, max_depth=3)

        for board_tensor, score in tree_data:
            board_data.append(board_tensor)
            score_data.append(torch.tensor(score, dtype=torch.float32))

            if len(board_data) >= batch_size:
                temp_file = f"{data_path}_part_{game_idx}.pt"
                torch.save((board_data, score_data), temp_file)
                temp_files.append(temp_file)
                board_data, score_data = [], []  # メモリ解放

    # 残りのデータを保存
    if board_data:
        temp_file = f"{data_path}_part_final.pt"
        torch.save((board_data, score_data), temp_file)
        temp_files.append(temp_file)

    # すべての部分ファイルを結合して最終ファイルを作成
    final_board_data, final_score_data = [], []
    for file in temp_files:
        if os.path.exists(file):
            boards, scores = torch.load(
                file, weights_only=True)  # weights_only=True を指定
            final_board_data.extend(boards)
            final_score_data.extend(scores)

    # 結合データを保存
    torch.save((final_board_data, final_score_data), data_path)

    # 一時ファイルを削除
    for file in temp_files:
        if os.path.exists(file):
            os.remove(file)


# データセット生成の実行
save_dataset(num_games=100, random_moves=10, data_path="othello_data.pt")

