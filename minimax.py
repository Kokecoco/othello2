from base import get_valid_moves, AI
import random


class MinimaxAI(AI):
    def __init__(self, evaluater=4):
        self.evaluater = evaluater

    def get_move(self, moves, board, stone):
        best_move = None
        best_value = float("-inf")

        for move in moves:
            new_board = self.make_move(board, move, stone)
            move_value = self.minimax(new_board, False, stone)

            if move_value > best_value:
                best_value = move_value
                best_move = move

        return best_move

    def minimax(self, board, is_maximizing, stone):
        opponent_stone = "B" if stone == "W" else "W"
        moves = get_valid_moves(opponent_stone, board)

        if not moves:
            if self.evaluater == 1:
                return self.evaluate_board1(board, stone)
            elif self.evaluater == 2:
                return self.evaluate_board2(board, stone)
            elif self.evaluater == 3:
                return self.evaluate_board3(board, stone)
            elif self.evaluater == 4:
                return self.evaluate_board4(board, stone)

        if is_maximizing:
            best_value = float("-inf")
            for move in moves:
                new_board = self.make_move(board, move, opponent_stone)
                best_value = max(best_value, self.minimax(new_board, False, stone))
            return best_value
        else:
            best_value = float("inf")
            for move in moves:
                new_board = self.make_move(board, move, stone)
                best_value = min(best_value, self.minimax(new_board, True, stone))
            return best_value

    def evaluate_board1(self, board, stone):
        # ボードの評価ロジックを実装
        # ここでは単純にランダムなスコアを返す例
        return random.randint(-10, 10)

    def evaluate_board3(self, board, stone):
        opponent_stone = "B" if stone == "W" else "W"
        stone_count = 0
        opponent_stone_count = 0
        for row in board:
            for cell in row:
                if cell == stone:
                    stone_count += 1
                elif cell == opponent_stone:
                    opponent_stone_count += 1

        score = stone_count - opponent_stone_count

        # 位置の価値を考慮する
        positional_value = [
            [100, -10, 10, 10, 10, 10, -10, 100],
            [-10, -20, 1, 1, 1, 1, -20, -10],
            [10, 1, 5, 5, 5, 5, 1, 10],
            [10, 1, 5, 5, 5, 5, 1, 10],
            [10, 1, 5, 5, 5, 5, 1, 10],
            [10, 1, 5, 5, 5, 5, 1, 10],
            [-10, -20, 1, 1, 1, 1, -20, -10],
            [100, -10, 10, 10, 10, 10, -10, 100],
        ]

        positional_score = 0
        for row in range(len(board)):
            for col in range(len(board[row])):
                if board[row][col] == stone:
                    positional_score += positional_value[row][col]
                elif board[row][col] == opponent_stone:
                    positional_score -= positional_value[row][col]

        total_score = score + positional_score
        return total_score

    def evaluate_board4(self, board, stone):
        opponent_stone = "B" if stone == "W" else "W"
        stone_count = 0
        opponent_stone_count = 0
        for row in board:
            for cell in row:
                if cell == stone:
                    stone_count += 1
                elif cell == opponent_stone:
                    opponent_stone_count += 1

        score = stone_count - opponent_stone_count

        # 位置の価値を考慮する
        positional_value = [
            [100, -10, 10, 10, 10, 10, -10, 100],
            [-10, -20, 1, 1, 1, 1, -20, -10],
            [10, 1, 5, 5, 5, 5, 1, 10],
            [10, 1, 5, 5, 5, 5, 1, 10],
            [10, 1, 5, 5, 5, 5, 1, 10],
            [10, 1, 5, 5, 5, 5, 1, 10],
            [-10, -20, 1, 1, 1, 1, -20, -10],
            [100, -10, 10, 10, 10, 10, -10, 100],
        ]

        positional_score = 0
        for row in range(len(board)):
            for col in range(len(board[row])):
                if board[row][col] == stone:
                    positional_score += positional_value[row][col]
                elif board[row][col] == opponent_stone:
                    positional_score -= positional_value[row][col]

        total_score = score + positional_score
        mobility_weight = 30
        mobility_score = 0
        mobility_score += len(get_valid_moves(stone, board)) * mobility_weight
        mobility_score -= len(get_valid_moves(opponent_stone, board)) * mobility_weight

        return total_score + mobility_score

    def evaluate_board2(self, board, stone):
        opponent_stone = "B" if stone == "W" else "W"
        stone_count = 0
        opponent_stone_count = 0
        for row in board:
            for cell in row:
                if cell == stone:
                    stone_count += 1
                elif cell == opponent_stone:
                    opponent_stone_count += 1

        score = stone_count - opponent_stone_count
        return score

    def make_move(self, board, move, stone):
        if 0 <= move[0] - 1 < len(board) and 0 <= move[1] - 1 < len(board[0]):
            new_board = [row[:] for row in board]  # 深いコピー
            new_board[move[0] - 1][move[1] - 1] = stone
            return new_board
        else:
            raise ValueError(f"Invalid move: {move}")
