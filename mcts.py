import cupy as cp
import random
from collections import defaultdict
from base import get_valid_moves, AI


class MonteCarloAI(AI):
    def __init__(self, evaluater=4, num_simulations=1000):
        self.evaluater = evaluater
        self.num_simulations = num_simulations

    def get_move(self, moves, board, stone):
        if not moves:
            return None

        # move_scoresとmove_countsをCuPy配列で定義
        move_scores = cp.zeros(len(moves), dtype=cp.int32)
        move_counts = cp.zeros(len(moves), dtype=cp.int32)

        # GPU上で並列処理を行うため、CuPy配列でインデックスを生成（int32型に明示的に変換）
        move_idx_array = cp.random.randint(
            0, len(moves), size=self.num_simulations, dtype=cp.int32)

        # シミュレーション結果のスコアを保存する配列を作成
        temp_scores = cp.zeros(self.num_simulations, dtype=cp.int32)

        # 各シミュレーションをGPUで並列に実行
        for i in range(self.num_simulations):
            move_idx = int(move_idx_array[i])
            move = moves[move_idx]
            new_board = self.make_move(board, move, stone)
            temp_scores[i] = self.run_simulation(new_board, stone)

        # GPU上でmove_scoresとmove_countsを集計
        cp.ElementwiseKernel(
            'int32 move_idx, int32 score',
            'raw int32 move_scores, raw int32 move_counts',
            'atomicAdd(&move_scores[move_idx], score); atomicAdd(&move_counts[move_idx], 1);'
        )(move_idx_array, temp_scores, move_scores, move_counts)

        # ベストの手を取得（ゼロ除算を避けるため、move_counts > 0 の場合のみ計算）
        best_move = moves[int(
            cp.argmax(cp.where(move_counts > 0, move_scores / move_counts, -cp.inf)))]
        return best_move

    def run_simulation(self, board, stone):
        current_stone = stone
        while True:
            moves = get_valid_moves(current_stone, board)
            if not moves:
                break
            move = random.choice(moves)
            board = self.make_move(board, move, current_stone)
            current_stone = 'B' if current_stone == 'W' else 'W'

        evaluator_funcs = {
            1: self.evaluate_board1,
            2: self.evaluate_board2,
            3: self.evaluate_board3,
            4: self.evaluate_board4,
        }
        return evaluator_funcs[self.evaluater](board, stone)

    def evaluate_board1(self, board, stone):
        return random.randint(-10, 10)

    def evaluate_board2(self, board, stone):
        opponent_stone = 'B' if stone == 'W' else 'W'
        stone_count = sum(row.count(stone) for row in board)
        opponent_stone_count = sum(row.count(opponent_stone) for row in board)
        return stone_count - opponent_stone_count

    def evaluate_board3(self, board, stone):
        opponent_stone = 'B' if stone == 'W' else 'W'
        stone_count = sum(row.count(stone) for row in board)
        opponent_stone_count = sum(row.count(opponent_stone) for row in board)
        score = stone_count - opponent_stone_count

        positional_value = [
            [100, -10, 10, 10, 10, 10, -10, 100],
            [-10, -20, 1, 1, 1, 1, -20, -10],
            [10, 1, 5, 5, 5, 5, 1, 10],
            [10, 1, 5, 5, 5, 5, 1, 10],
            [10, 1, 5, 5, 5, 5, 1, 10],
            [10, 1, 5, 5, 5, 5, 1, 10],
            [-10, -20, 1, 1, 1, 1, -20, -10],
            [100, -10, 10, 10, 10, 10, -10, 100]
        ]

        positional_score = sum(
            positional_value[row][col] if cell == stone else -
            positional_value[row][col]
            for row, board_row in enumerate(board)
            for col, cell in enumerate(board_row) if cell in [stone, opponent_stone]
        )

        return score + positional_score

    def evaluate_board4(self, board, stone):
        opponent_stone = 'B' if stone == 'W' else 'W'
        stone_count = sum(row.count(stone) for row in board)
        opponent_stone_count = sum(row.count(opponent_stone) for row in board)
        score = stone_count - opponent_stone_count

        positional_value = [
            [100, -10, 10, 10, 10, 10, -10, 100],
            [-10, -20, 1, 1, 1, 1, -20, -10],
            [10, 1, 5, 5, 5, 5, 1, 10],
            [10, 1, 5, 5, 5, 5, 1, 10],
            [10, 1, 5, 5, 5, 5, 1, 10],
            [10, 1, 5, 5, 5, 5, 1, 10],
            [-10, -20, 1, 1, 1, 1, -20, -10],
            [100, -10, 10, 10, 10, 10, -10, 100]
        ]

        positional_score = sum(
            positional_value[row][col] if cell == stone else -
            positional_value[row][col]
            for row, board_row in enumerate(board)
            for col, cell in enumerate(board_row) if cell in [stone, opponent_stone]
        )

        mobility_weight = 30
        mobility_score = (len(get_valid_moves(stone, board)) -
                          len(get_valid_moves(opponent_stone, board))) * mobility_weight
        return score + positional_score + mobility_score

    def make_move(self, board, move, stone):
        if 0 <= move[0]-1 < len(board) and 0 <= move[1]-1 < len(board[0]):
            new_board = [row[:] for row in board]  # 深いコピー
            new_board[move[0]-1][move[1]-1] = stone
            return new_board
        else:
            raise ValueError(f"Invalid move: {move}")
