import random


class Board:
    W = 1  # 白
    B = -1  # 黒
    E = 0  # 空
    DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

    def __init__(self, init_board=None):
        if init_board is None:
            self.board = [
                [self.E, self.E, self.E, self.E, self.E, self.E, self.E, self.E],
                [self.E, self.E, self.E, self.E, self.E, self.E, self.E, self.E],
                [self.E, self.E, self.E, self.E, self.E, self.E, self.E, self.E],
                [self.E, self.E, self.E, self.W, self.B, self.E, self.E, self.E],
                [self.E, self.E, self.E, self.B, self.W, self.E, self.E, self.E],
                [self.E, self.E, self.E, self.E, self.E, self.E, self.E, self.E],
                [self.E, self.E, self.E, self.E, self.E, self.E, self.E, self.E],
                [self.E, self.E, self.E, self.E, self.E, self.E, self.E, self.E],
            ]
        else:
            self.board = init_board

    def set_stone_on_board(self, x, y, stone):
        stones = self.get_flippable_stones(x, y, stone)
        for stone_pos in stones:
            self.flip_stone(stone_pos[0], stone_pos[1])

        if not (0 <= x < 8 and 0 <= y < 8):
            return None
        elif self.board[x][y] == self.W or self.board[x][y] == self.B:
            return None
        else:
            self.board[x][y] = stone
            return stone

    def flip_stone(self, x, y):
        if not (0 <= x < 8 and 0 <= y < 8):
            return None
        elif self.board[x][y] == self.E:
            return None
        else:
            self.board[x][y] *= -1

    def print_board(self):
        print("　１２３４５６７８")
        for i in range(8):
            print(han_to_zen(i + 1), end="")
            for j in range(8):
                print(encode_number(self.board[i][j]), end="")
            print()

    def get_flippable_stones(self, x, y, stone):
        if not (0 <= x < 8 and 0 <= y < 8):
            return []
        if self.board[x][y] != self.E:
            return []

        opponent = self.W if stone == self.B else self.B
        flippable_stones = []

        for d in self.DIRECTIONS:
            temp_stones = []
            nx, ny = x + d[0], y + d[1]

            while 0 <= nx < 8 and 0 <= ny < 8:
                if self.board[nx][ny] == opponent:
                    temp_stones.append((nx, ny))
                elif self.board[nx][ny] == stone:
                    if temp_stones:
                        flippable_stones.extend(temp_stones)
                    break
                else:
                    break

                nx += d[0]
                ny += d[1]

        return flippable_stones

    def get_valid_moves(self, stone):
        valid_moves = []
        for x in range(8):
            for y in range(8):
                if self.board[x][y] == self.E and self.get_flippable_stones(
                    x, y, stone
                ):
                    valid_moves.append((x + 1, y + 1))  # 1から始まる座標系に変換
        return valid_moves

    def count_stones(self, stone):
        counts = 0
        for i in range(8):
            counts += self.board[i].count(stone)
        return counts


# 補助関数


def han_to_zen(han):
    return "１２３４５６７８"[han - 1]


def encode_number(num):
    if num == Board.W:
        return "白"
    elif num == Board.B:
        return "黒"
    elif num == Board.E:
        return "　"


def get_valid_moves(stone, board):
    valid_moves = []
    for x in range(8):
        for y in range(8):
            if board[x][y] == Board.E and get_flippable_stones(x, y, stone, board):
                valid_moves.append((x + 1, y + 1))  # 1から始まる座標系に変換
    return valid_moves


def get_flippable_stones(x, y, stone, board):
    if not (0 <= x < 8 and 0 <= y < 8):
        return []
    if board[x][y] != Board.E:
        return []

    opponent = Board.W if stone == Board.B else Board.B
    flippable_stones = []

    for d in Board.DIRECTIONS:
        temp_stones = []
        nx, ny = x + d[0], y + d[1]
        while 0 <= nx < 8 and 0 <= ny < 8:
            if board[nx][ny] == opponent:
                temp_stones.append((nx, ny))
            elif board[nx][ny] == stone:
                if temp_stones:
                    flippable_stones.extend(temp_stones)
                break
            else:
                break

            nx += d[0]
            ny += d[1]
    return flippable_stones


def play_user_by_user():
    game = Board(None)
    game.print_board()
    for _ in range(30):
        B = input("B>").split(",")
        while (
            len(game.get_flippable_stones(int(B[0]) - 1, int(B[1]) - 1, Board.B)) == 0
        ):
            print("置くことが可能な手を指定してください。")
            B = input("B>").split(",")
        game.set_stone_on_board(int(B[0]) - 1, int(B[1]) - 1, Board.B)
        game.print_board()

        W = input("W>").split(",")
        while (
            len(game.get_flippable_stones(int(W[0]) - 1, int(W[1]) - 1, Board.W)) == 0
        ):
            print("置くことが可能な手を指定してください。")
            W = input("W>").split(",")
        game.set_stone_on_board(int(W[0]) - 1, int(W[1]) - 1, Board.W)
        game.print_board()


def play_user_by_cpu(cpu):
    turn = Board.B
    game = Board(None)
    game.print_board()
    while len(game.get_valid_moves(turn)):
        if turn == Board.B:
            move = input("B>").split(",")
            while (
                len(move) != 2
                or move[0] == ""
                or move[1] == ""
                or len(
                    game.get_flippable_stones(int(move[0]) - 1, int(move[1]) - 1, turn)
                )
                == 0
            ):
                print("置くことが可能な手を指定してください。")
                move = input("B>").split(",")
        else:
            moves = game.get_valid_moves(turn)
            move = cpu.get_move(moves, game.board, turn)
        game.set_stone_on_board(int(move[0]) - 1, int(move[1]) - 1, turn)
        game.print_board()
        turn *= -1
        print()
    print(
        "B:" + str(game.count_stones(Board.B)) + "W:" + str(game.count_stones(Board.W))
    )


def play_cpu_by_cpu(cpu1, cpu2, do_print):
    turn = Board.B
    game = Board(None)
    if do_print:
        game.print_board()

    while True:
        moves = game.get_valid_moves(turn)
        if len(moves) == 0:
            if len(game.get_valid_moves(-turn)) == 0:
                break  # 両方のプレイヤーが置けない場合、ゲーム終了
            else:
                turn *= -1  # パス
                continue
        if turn == Board.B:
            move = cpu1.get_move(moves, game.board, turn)
        else:
            move = cpu2.get_move(moves, game.board, turn)
        game.set_stone_on_board(int(move[0]) - 1, int(move[1]) - 1, turn)
        if do_print:
            game.print_board()
        turn *= -1
        if do_print:
            print()

    B_stones = game.count_stones(Board.B)
    W_stones = game.count_stones(Board.W)

    if do_print:
        print("B:" + str(B_stones) + ", W:" + str(W_stones))

    if W_stones < B_stones:
        winner = "B"
    elif W_stones > B_stones:
        winner = "W"
    else:
        winner = ""

    return winner


class AI:
    def get_move(self, moves, board, stone):
        return moves[0]


class RandomAI(AI):
    def get_move(self, moves, board, stone):
        return random.choice(moves)


class MaxAI(AI):
    def get_move(self, moves, board, stone):
        max_move = []
        max_flippable = 0
        for move in moves:
            flippable = len(
                get_flippable_stones(move[0] - 1, move[1] - 1, stone, board)
            )
            if max_flippable < flippable:
                max_flippable = flippable
                max_move = move
        return max_move


class CornerAI(AI):
    def get_move(self, moves, board, stone):
        corners = [(0, 0), (0, 7), (7, 0), (7, 7)]
        moves = get_valid_moves(stone, board)
        corner_moves = [move for move in moves if move in corners]
        if corner_moves:
            return random.choice(corner_moves)
        return random.choice(moves) if moves else None
