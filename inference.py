# inference.py
import torch
from base import AI, get_valid_moves
from initial_training import BoardEvaluator, board_to_tensor
from self_play_training import make_move


class DeepLearningAI(AI):
    def __init__(self, model_path):
        self.model = BoardEvaluator()
        self.model.load_state_dict(torch.load(model_path, weights_only=True))
        self.model.eval()

    def get_move(self, moves, board, stone):
        best_move = None
        best_value = float("-inf")

        for move in moves:
            new_board = make_move(board, move, stone)
            board_tensor = board_to_tensor(new_board, stone)
            move_value = self.model(board_tensor).item()

            if move_value > best_value:
                best_value = move_value
                best_move = move

        return best_move


if __name__ == "__main__":
    ai = DeepLearningAI("final_model_weights.pth")
    board = [[0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, -1, 1, 0, 0, 0, 0],
             [0, 0, 1, -1, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0]]
    moves = get_valid_moves(1, board)
    best_move = ai.get_move(moves, board, 1)
    print(f"AIの選んだ手: {best_move}")
