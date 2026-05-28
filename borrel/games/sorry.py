import copy

import numpy as np
import pandas as pd

# Is multiplied by nr of players for total board length
BASE_BOARD_LENGTH = 5


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
