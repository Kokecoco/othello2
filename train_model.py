import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from base import Board  # Boardクラスをbase.pyからインポート

# モデル定義


class OthelloEvaluationModel(nn.Module):
    def __init__(self):
        super(OthelloEvaluationModel, self).__init__()
        # 入力層 (8x8 = 64) -> 隠れ層1 (128) -> 隠れ層2 (64) -> 出力層 (1)
        self.fc1 = nn.Linear(8 * 8, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = x.view(-1, 8 * 8)      # 8x8ボードを1次元に平坦化
        x = torch.relu(self.fc1(x))  # 隠れ層1
        x = torch.relu(self.fc2(x))  # 隠れ層2
        x = self.fc3(x)             # 出力層
        return x

# 学習関数


def train_model(data_path, epochs=10, batch_size=32, learning_rate=0.001):
    # データの読み込み
    board_tensors, score_tensors = torch.load(data_path)
    board_tensors = board_tensors.float()  # データ型をfloatに変換
    score_tensors = score_tensors.float()  # データ型をfloatに変換

    # データセットとデータローダーの作成
    dataset = TensorDataset(board_tensors, score_tensors)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # モデル、損失関数、オプティマイザの準備
    model = OthelloEvaluationModel()
    criterion = nn.MSELoss()  # 評価スコアの予測なので回帰タスクとしてMSEを使用
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 学習ループ
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for boards, scores in dataloader:
            # 順伝播
            predictions = model(boards)
            loss = criterion(predictions, scores)

            # 逆伝播とパラメータ更新
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(
            f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(dataloader)}")

    # 学習後のモデルを保存
    torch.save(model.state_dict(), "othello_evaluation_model.pt")
    print("学習完了！モデルは 'othello_evaluation_model.pt' に保存されました。")


# メイン実行部
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print(
            "使用法: python train_model.py <data_path> [epochs] [batch_size] [learning_rate]")
    else:
        data_path = sys.argv[1]
        epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 10
        batch_size = int(sys.argv[3]) if len(sys.argv) > 3 else 32
        learning_rate = float(sys.argv[4]) if len(sys.argv) > 4 else 0.001

        train_model(data_path, epochs, batch_size, learning_rate)
