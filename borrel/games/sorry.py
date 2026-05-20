import copy

import numpy as np
import pandas as pd

# Is multiplied by nr of players for total board length
BASE_BOARD_LENGTH = 5


"""
This is a slightly modified version of the classic game of Sorry! (Mens erger je niet!)

Every turn you will get a pandas dataframe with the current state of the board
It will have a row for each board space
It will have a column called "home" which will hold None or the name of the player whose home it is (e.g. "Mees" or "Ivo")
- Homes are evenly space apart, every 5 spaces
It will have a column called "space" which will hold either None (empty space) or a piece
Pieces are named as "playername_piecenum" (e.g. "Ivo_1", "Jan_2")
Each player has 4 pieces, numbered 1-4
Pieces move down this board. Pieces placed on the board will be placed on the first space after their home space.
The board is circular, so after the last space it goes back to the first space.
A piece that lands on its home space is considered finished and is removed from the board.
The first player to get all 4 pieces finished wins the game.

Additionally you will get a dictionary with some info on the game, which will have the following keys:
- "pieces_at_home": a dictionary with player names as keys and a list of piece numbers at home as values
- "pieces_finished": a dictionary with player names as keys and a list of piece numbers that have finished as values

Additionally you will receive your current dice roll (a random number in the range [1, 6])
- if this number is a 6, you can choose to move a piece from home to the starting position (if you have any pieces at home) or move a piece on the board
- if this number is not a 6, you must move a piece on the board (if you have any pieces on the board)

You must return the number of the piece you want to move (1-4). This piece will then be moved according to the rules of the game.

Moving rules:
- If you move a piece from home to the starting position, it will be placed on the first space of the board after your home space
    - This means that if you overshoot your home, you will just continue moving!
- If you move a piece on the board, it will move forward a number of spaces equal to the dice roll
- If a piece lands on a space occupied by another piece, the other piece is sent back to its home

Examples (with only 2 players, Mark and Dirk):
| index | home  | space |
| 0     | Mark  | None  |
| 1     | None  | None  |
| 2     | None  | None  |
| 3     | None  | None  |
| 4     | None  | None  |
| 5     | Dirk  | None  |
| 6     | None  | None  |
| 7     | None  | None  |
| 8     | None  | None  |
| 9     | None  | None  |
info dict:
{
    "pieces_at_home": {"Mark": [1, 2, 3, 4], "Dirk": [1, 2, 3, 4]},
    "pieces_finished": {"Mark": [], "Dirk": []},
}

Mark now rolls a 6 and decides to move piece 1 from home to the starting position. The board now looks like this:
| index | home  | space |
| 0     | Mark  | None  |
| 1     | None  | Mark_1|
| 2     | None  | None  |
| 3     | None  | None  |
| 4     | None  | None  |
| 5     | Dirk  | None  |
| 6     | None  | None  |
| 7     | None  | None  |
| 8     | None  | None  |
| 9     | None  | None  |
info dict:
{
    "pieces_at_home": {"Mark": [2, 3, 4], "Dirk": [1, 2, 3, 4]},
    "pieces_finished": {"Mark": [], "Dirk": []},
}

Several turn later the board looks like this:
| index | home  | space |
| 0     | Mark  | None  |
| 1     | None  | None  |
| 2     | None  | None  |
| 3     | None  | None  |
| 4     | None  | None  |
| 5     | Dirk  | Mark_1|
| 6     | None  | None  |
| 7     | None  | None  |
| 8     | None  | Dirk_1|
| 9     | None  | None  |
info dict:
{
    "pieces_at_home": {"Mark": [2, 3, 4], "Dirk": [1, 2, 3, 4]},
    "pieces_finished": {"Mark": [], "Dirk": []},
}

Mark now rolls a 3 and decides to move piece 1. The board now looks like this:
| index | home  | space |
| 0     | Mark  | None  |
| 1     | None  | None  |
| 2     | None  | None  |
| 3     | None  | None  |
| 4     | None  | None  |
| 5     | Dirk  | None  |
| 6     | None  | None  |
| 7     | None  | None  |
| 8     | None  | Mark_1|
| 9     | None  | None  |
info dict:
{
    "pieces_at_home": {"Mark": [2, 3, 4], "Dirk": [1, 2, 3, 4]},
    "pieces_finished": {"Mark": [], "Dirk": []},
}

Dirks piece is sent back to his home.
If mark now rolls a 2, he can move piece 1, by returning the integer 1, to the home space and finish it.
If mark now rolls a 4, he can move piece 1, by returning the integer 1, overshooting his home. The piece will end up on index 3
If mark now rolls a 6, he can either:
- return the integer 1 to move piece 1 past his home, ending up on index 5
- return integer 2, 3 or 4 to move a new piece from home to the starting position, ending up on index 1

Say that mark rolls a 2 and decides to move piece 1, then the board will look like this:
| index | space | home  |
| 0     | None  | Mark  |
| 1     | None  | None  |
| 2     | None  | None  |
| 3     | None  | None  |
| 4     | None  | None  |
| 5     | None  | Dirk  |
| 6     | None  | None  |
| 7     | None  | None  |
| 8     | None  | None  |
| 9     | None  | None  |
info dict:
{
    "pieces_at_home": {"Mark": [2, 3, 4], "Dirk": [1, 2, 3, 4]},
    "pieces_finished": {"Mark": [1], "Dirk": []},
}

If you perform an illegal move, a random piece of yours is removed from the board and sent back to your home.
A move is illegal if it refers to a piece that cannot be moved or placed on the board
If no pieces can be removed from the board, nothing happens

After every full round of turns the game checks if any player has won by getting all 4 pieces finished. This means two players can tie and both win.
"""


class Sorry:
    def __init__(self, players: list, render: bool = False):
        self.board = None
        self.info = {
            "pieces_at_home": {player.name: [1, 2, 3, 4] for player in players},
            "pieces_finished": {player.name: [] for player in players},
        }

        # We randomize the order of the players to randomize turn order and home positions
        np.random.shuffle(players)
        self.players = players
        self.render = render

    def init_board(self):
        board_length = BASE_BOARD_LENGTH * len(self.players)
        self.board = pd.DataFrame(
            {
                "home": [None] * board_length,
                "space": [None] * board_length,
            }
        )
        for i, player in enumerate(self.players):
            home_index = (i * BASE_BOARD_LENGTH) % board_length
            self.board.loc[home_index, "home"] = player.name

    def render_state(self, dice_roll: int, turn_of_player: str):
        print(f"Turn of {turn_of_player}, dice roll: {dice_roll}")
        print(self.board)
        print(f"pieces at home: {self.info['pieces_at_home']}")
        print(f"pieces finished: {self.info['pieces_finished']}")

    def is_move_legal(self, player, dice_roll: int, move: int) -> bool:
        if not isinstance(move, int):
            return False

        if move not in [1, 2, 3, 4]:
            return False

        if dice_roll != 6:
            # If the move is not a 6, the piece must be on the board
            piece_name = f"{player}_{move}"
            if piece_name not in self.board["space"].values:
                return False
        else:
            # If the move is a 6, the piece can either be on the board or at home
            piece_name = f"{player}_{move}"
            if (
                piece_name not in self.board["space"].values
                and move not in self.info["pieces_at_home"][player]
            ):
                return False

        return True

    def play_turn(self):
        for player in self.players:
            dice_roll = np.random.randint(1, 7)

            if self.render:
                self.render_state(dice_roll, player.name)

            move = player.sorry(
                copy.deepcopy(self.board), copy.deepcopy(self.info), dice_roll
            )

            if self.render:
                print(f"{player.name} decides to move piece {move}")

            if not self.is_move_legal(player.name, dice_roll, move):
                # Illegal move, send a random piece back to home
                pieces_on_board = self.board[
                    self.board["space"].str.startswith(player.name, na=False)
                ]
                if not pieces_on_board.empty:
                    piece_to_remove = pieces_on_board.sample(1)
                    self.board.loc[piece_to_remove.index, "space"] = None
                    piece_num = int(piece_to_remove["space"].values[0].split("_")[-1])
                    self.info["pieces_at_home"][player.name].append(piece_num)
                continue

            # Move is legal, perform the move
            # Find the piece to move
            piece_name = f"{player.name}_{move}"
            piece_idx = self.board[self.board["space"] == piece_name].index
            if piece_idx.empty:
                # The piece is not on the board, it must be at home
                start_idx = (
                    self.board[self.board["home"] == player.name].index[0] + 1
                ) % len(self.board)
                self.board.loc[start_idx, "space"] = piece_name
                self.info["pieces_at_home"][player.name].remove(move)
            else:
                # The piece is on the board, move it forward
                current_idx = piece_idx[0]
                new_idx = (current_idx + dice_roll) % len(self.board)

                # Check if the piece lands on another piece
                if self.board.loc[new_idx, "space"] is not None:
                    other_piece_name = self.board.loc[new_idx, "space"]
                    other_player_name = "_".join(other_piece_name.split("_")[:-1])
                    other_piece_num = int(other_piece_name.split("_")[-1])
                    self.info["pieces_at_home"][other_player_name].append(
                        other_piece_num
                    )

                # Remove the piece from its current position
                self.board.loc[current_idx, "space"] = None

                # Place the piece on the new position or mark it as finished
                if self.board.loc[new_idx, "home"] == player.name:
                    # The piece has finished
                    self.info["pieces_finished"][player.name].append(move)
                else:
                    # Place the piece on the new space
                    self.board.loc[new_idx, "space"] = piece_name

    def reset(self):
        self.init_board()
        for player in self.players:
            self.info["pieces_at_home"][player.name] = [1, 2, 3, 4]
            self.info["pieces_finished"][player.name] = []

    def play_game(self, n_games: int = 10):
        scores = {player.name: 0 for player in self.players}

        for _ in range(n_games):
            self.reset()
            while True:
                self.play_turn()
                for player in self.players:
                    if len(self.info["pieces_finished"][player.name]) == 4:
                        scores[player.name] += 1
                        break
                else:
                    continue
                break
