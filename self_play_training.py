# self_play_training.py
import torch
import torch.nn as nn
import torch.optim as optim
import random
from base import get_valid_moves
from initial_training import BoardEvaluator, board_to_tensor, generate_evaluation_score


def make_move(board, move, stone):
    new_board = [row[:] for row in board]
    new_board[move[0] - 1][move[1] - 1] = stone
    return new_board


def train_with_self_play(model, optimizer, criterion, num_epochs=1000):
    model.train()
    for epoch in range(num_epochs):
        optimizer.zero_grad()

        # 初期のボード状態を設定（0は空き、1と-1は各プレイヤーの石）
        board = [[0 for _ in range(8)] for _ in range(8)]
        board[3][3], board[4][4] = 1, 1
        board[3][4], board[4][3] = -1, -1
        stone = 1

        for _ in range(30):
            moves = get_valid_moves(stone, board)
            if not moves:
                break

            best_move = None
            best_value = float("-inf")
            for move in moves:
                new_board = make_move(board, move, stone)
                board_tensor = board_to_tensor(new_board, stone)
                move_value = model(board_tensor).item()
                if move_value > best_value:
                    best_value = move_value
                    best_move = move

            if best_move:
                board = make_move(board, best_move, stone)
            stone = -stone

        final_score = generate_evaluation_score(board, 1)
        final_tensor = board_to_tensor(board, 1)
        prediction = model(final_tensor)
        target = torch.tensor([final_score], dtype=torch.float32).unsqueeze(0)
        loss = criterion(prediction, target)
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(
                f"Self-Play Training Epoch {epoch}/{num_epochs}, Loss: {loss.item()}")

        if float(loss.item()) < 0.001:
            print(
                f"Early stopping at epoch {epoch} as loss has reached below 0.001.")
            break

    torch.save(model.state_dict(), "final_model_weights.pth")
    print("強化学習済みモデルを保存しました。")


if __name__ == "__main__":
    model = BoardEvaluator()
    model.load_state_dict(torch.load("initial_model_weights.pth"))
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    train_with_self_play(model, optimizer, criterion)
