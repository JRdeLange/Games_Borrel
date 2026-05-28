from typing import Literal, cast

import numpy as np
import pandas as pd


class Mees:
    def __init__(self, name: str = "mees"):
        # NOTE: DO NOT TOUCH!
        self.name = name  # NOTE: DO NOT TOUCH!
        # NOTE: DO NOT TOUCH!

        # Feel free to store whatever you want here.
        # Each full run of a game will use a fresh instance of this class.
        # So for for example battleship, a new instance will be created for each game, but not for each turn.

        self._rps_state: dict | None = None

    def tron(self, grid: np.ndarray) -> Literal["up", "down", "left", "right"]:
        """
        This game is inspired by the lightcycles in the movie Tron.
        These lightcycle bikes leave a wall behind them as they move, which kills you if you run into them.
        This implementation is grid-based; think multiplayer snake without food and tails that do not disappear.

        You receive a grid representing the current state of the game.
        This grid is a 2D array of characters, with the following characters:
        - " " represents an empty cell
        - "o" represents walls created by you
        - ">" or "<" or "^" or "v" represents your bike, depending on the direction you are going
        - "x" represents walls created by the opponent
        - "X" represents the opponent's bike

        Example input (5x5 grid):

        +++++++++++++
        +           +
        +     X x   +
        +   o   x   +
        +   o       +
        +   v       +
        +++++++++++++

        You die if you hit a wall, a bike, or leave the grid. So in this example,
        you are about to run into the bottom wall if you do not move right or left.

        You win if the opponent dies.

        Both players move at the same time.
        - If you both move into the same cell, you both die.
        - If you both die at the same time, it is a tie.

        Write a function that returns "up", "down", "left", or "right" based on the current state of the game.
        If you return the opposite of the direction you are currently going, you will keep going in the direction you currently are.

        The grid size and starting positions are always the same between games (7x7 and [(3, 2), (3, 4)])

        Good luck and be happy you are not actually trapped in a computer forced to compete to the death!
        """
        from collections import deque

        head_chars = {">": (0, 1), "<": (0, -1), "^": (-1, 0), "v": (1, 0)}
        my_pos = opp_pos = my_dir = None
        rows, cols = grid.shape

        for r in range(rows):
            for c in range(cols):
                ch = grid[r, c]
                if ch in head_chars:
                    my_pos, my_dir = (r, c), head_chars[ch]
                elif ch == "X":
                    opp_pos = (r, c)

        DIRS = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}
        opposite = (-my_dir[0], -my_dir[1])

        def bfs_from(start):
            dist = {start: 0}
            q = deque([start])
            while q:
                r, c = q.popleft()
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = r + dr, c + dc
                    npos = (nr, nc)
                    if (
                        0 <= nr < rows
                        and 0 <= nc < cols
                        and npos not in dist
                        and grid[nr, nc] == " "
                    ):
                        dist[npos] = dist[(r, c)] + 1
                        q.append(npos)
            return dist

        opp_dist = bfs_from(opp_pos) if opp_pos is not None else {}

        best_move = None
        best_score = (-1, -1)

        for move_name, (dr, dc) in DIRS.items():
            if (dr, dc) == opposite:
                continue
            nr, nc = my_pos[0] + dr, my_pos[1] + dc
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue
            if grid[nr, nc] != " ":
                continue

            my_dist = bfs_from((nr, nc))
            my_territory = sum(
                1
                for pos, d in my_dist.items()
                if pos not in opp_dist or d <= opp_dist[pos]
            )
            opp_territory = sum(
                1
                for pos, d in opp_dist.items()
                if pos not in my_dist or d < my_dist[pos]
            )
            score = (my_territory, -opp_territory)
            if score > best_score:
                best_score = score
                best_move = move_name

        if best_move is None:
            for move_name, (dr, dc) in DIRS.items():
                if (dr, dc) != opposite:
                    return cast(Literal["up", "down", "left", "right"], move_name)
        return cast(Literal["up", "down", "left", "right"], best_move)

    def dots_and_lines(
        self,
        horizontal_lines: np.ndarray,
        vertical_lines: np.ndarray,
        box_owners: np.ndarray,
        history: pd.DataFrame,
    ) -> dict:
        """
        Classic dots-and-boxes style game for two players.

        How the board is represented:
        - Dots are implicit: there are (size + 1) x (size + 1) dots.
        - horizontal_lines: 2D bool array of shape (size + 1, size).
          True means the horizontal edge at that position has been drawn.
        - vertical_lines: 2D bool array of shape (size, size + 1).
          True means the vertical edge at that position has been drawn.
        - box_owners: 2D array of shape (size, size), where each cell is either
          None or the name of the player that completed that box.

        Turn flow:
        - Current player draws a line.
        - If the line completes one or more boxes, those boxes are scored and the
          same player gets another turn.
        - Otherwise the turn passes to the opponent.
        - When all lines are drawn, the player with the most boxes wins.

        You also receive a history dataframe with the following columns:
        - player: the name of the player who drew the line
        - orientation: "h" (horizontal) or "v" (vertical)
        - row: the row index of the line
        - col: the column index of the line
        - boxes_completed: the number of boxes completed by this move
        - <player1>_score: score of player 1 after this move
        - <player2>_score: score of player 2 after this move
        - lines_remaining: the number of lines still to be drawn

        Invalid move behavior:
        - If your function crashes, returns an invalid format, or picks an
          already-drawn line, a random legal move is played for you instead
          and you will be informed via a printed message.

        Write a function that returns a move in this format:
        - {"orientation": "h"|"v", "row": int, "col": int}

        Example of how a board is built up (5x5 boxes, so 6x6 dots):

        Start — empty board:
        .   .   .   .   .   .

        .   .   .   .   .   .

        .   .   .   .   .   .

        .   .   .   .   .   .

        .   .   .   .   .   .

        .   .   .   .   .   .

        After h(row=0, col=0) — top edge of top-left box:
        .---.   .   .   .   .

        .   .   .   .   .   .

        .   .   .   .   .   .

        .   .   .   .   .   .

        .   .   .   .   .   .

        .   .   .   .   .   .

        After v(row=0, col=0) and v(row=0, col=1) — left and right edges:
        .---.   .   .   .   .
        |   |
        .   .   .   .   .   .

        .   .   .   .   .   .

        .   .   .   .   .   .

        .   .   .   .   .   .

        .   .   .   .   .   .

        After h(row=1, col=0) — bottom edge completes the box, scored to player A:
        .---.   .   .   .   .
        | A |
        .---.   .   .   .   .

        .   .   .   .   .   .

        .   .   .   .   .   .

        .   .   .   .   .   .

        .   .   .   .   .   .

        The board size is always 5x5 (so 6x6 dots, 6x5 horizontal lines, 5x6 vertical lines).

        Good luck connecting those dots!
        """
        size = horizontal_lines.shape[1]

        def box_sides(r, c):
            return (
                int(horizontal_lines[r, c])
                + int(horizontal_lines[r + 1, c])
                + int(vertical_lines[r, c])
                + int(vertical_lines[r, c + 1])
            )

        def adjacent_boxes(orient, r, c):
            if orient == "h":
                return ([(r - 1, c)] if r > 0 else []) + ([(r, c)] if r < size else [])
            else:
                return ([(r, c - 1)] if c > 0 else []) + ([(r, c)] if c < size else [])

        def would_complete(orient, r, c):
            return any(
                box_sides(br, bc) == 3 for br, bc in adjacent_boxes(orient, r, c)
            )

        def would_create_3sided(orient, r, c):
            return any(
                box_sides(br, bc) == 2 for br, bc in adjacent_boxes(orient, r, c)
            )

        def simulate_opp_capture(orient, r, c):
            hl = horizontal_lines.copy()
            vl = vertical_lines.copy()
            if orient == "h":
                hl[r, c] = True
            else:
                vl[r, c] = True

            def sides(br, bc):
                return (
                    int(hl[br, bc])
                    + int(hl[br + 1, bc])
                    + int(vl[br, bc])
                    + int(vl[br, bc + 1])
                )

            total = 0
            changed = True
            while changed:
                changed = False
                for br in range(size):
                    for bc in range(size):
                        if sides(br, bc) == 3:
                            if not hl[br, bc]:
                                hl[br, bc] = True
                            elif not hl[br + 1, bc]:
                                hl[br + 1, bc] = True
                            elif not vl[br, bc]:
                                vl[br, bc] = True
                            else:
                                vl[br, bc + 1] = True
                            total += 1
                            changed = True
            return total

        legal_h = [
            (r, c)
            for r in range(size + 1)
            for c in range(size)
            if not horizontal_lines[r, c]
        ]
        legal_v = [
            (r, c)
            for r in range(size)
            for c in range(size + 1)
            if not vertical_lines[r, c]
        ]
        all_moves = [("h", r, c) for r, c in legal_h] + [
            ("v", r, c) for r, c in legal_v
        ]

        def immediate_closed(orient, r, c, hl=horizontal_lines, vl=vertical_lines):
            hl2, vl2 = hl.copy(), vl.copy()
            if orient == "h":
                hl2[r, c] = True
            else:
                vl2[r, c] = True

            def s(br, bc):
                return (
                    int(hl2[br, bc])
                    + int(hl2[br + 1, bc])
                    + int(vl2[br, bc])
                    + int(vl2[br, bc + 1])
                )

            return sum(1 for br, bc in adjacent_boxes(orient, r, c) if s(br, bc) == 4)

        def opp_best_take(hl, vl):
            best = 0
            for r2, c2 in [
                (r, c) for r in range(size + 1) for c in range(size) if not hl[r, c]
            ]:
                best = max(best, immediate_closed("h", r2, c2, hl, vl))
            for r2, c2 in [
                (r, c) for r in range(size) for c in range(size + 1) if not vl[r, c]
            ]:
                best = max(best, immediate_closed("v", r2, c2, hl, vl))
            return best

        # 1. Among scoring moves, pick the one that minimizes opponent counter-play
        scoring = [(o, r, c) for o, r, c in all_moves if would_complete(o, r, c)]
        if scoring:
            best_move = None
            best_score = -1e18
            for o, r, c in scoring:
                now = immediate_closed(o, r, c)
                hl2 = horizontal_lines.copy()
                vl2 = vertical_lines.copy()
                if o == "h":
                    hl2[r, c] = True
                else:
                    vl2[r, c] = True
                opp_take = opp_best_take(hl2, vl2)
                thirds = int(would_create_3sided(o, r, c))
                score = (
                    500.0 * now
                    - 180.0 * opp_take
                    - 40.0 * thirds
                    + float(np.random.uniform(0, 1e-3))
                )
                if score > best_score:
                    best_score = score
                    best_move = {"orientation": o, "row": r, "col": c}
            return best_move

        # 2. Safe moves: pick the one that minimizes opponent's next-turn gain
        safe = [(o, r, c) for o, r, c in all_moves if not would_create_3sided(o, r, c)]
        if not safe:
            # All moves dangerous — minimize chain given away
            best = min(all_moves, key=lambda m: simulate_opp_capture(*m))
            return {"orientation": best[0], "row": best[1], "col": best[2]}

        best_move = None
        best_score = -1e18
        for o, r, c in safe:
            hl2 = horizontal_lines.copy()
            vl2 = vertical_lines.copy()
            if o == "h":
                hl2[r, c] = True
            else:
                vl2[r, c] = True
            opp_take = opp_best_take(hl2, vl2)
            thirds = int(would_create_3sided(o, r, c))
            center_dist = (
                abs(r - size / 2) + abs(c - (size - 1) / 2)
                if o == "h"
                else abs(r - (size - 1) / 2) + abs(c - size / 2)
            )
            score = (
                -260.0 * opp_take
                - 70.0 * thirds
                - 2.0 * center_dist
                + float(np.random.uniform(0, 1e-3))
            )
            if score > best_score:
                best_score = score
                best_move = {"orientation": o, "row": r, "col": c}
        return best_move

    def sorry(self, board: pd.DataFrame, info: dict, dice_roll: int) -> int:
        """
        In this game you pick between two options: "A" or "B".
        You are playing against all other players at once.
        Each round point is granted to all players who choose the option chosen by the fewest nr of players.

        You receive a pandas dataframe representing the history of the game.
        The dataframe has a column for each player, and in each row the choice of the player in that round:

          dummy1 dummy2 dummy3  rounds_remaining
        0      B      B      A               4.0
        1      A      A      A               3.0
        2      A      B      A               2.0
        3      B      A      B               1.0
        4      A      B      B               0.0

        These are dummy names and will be e.g. "ivo" "do" "carlos" in the scoring round.
        I advise you to deal with these names dynamically, as they are not guaranteed to all be present.

        The score in the example above would be:
        {'dummy1': 1, 'dummy2': 2, 'dummy3': 1}

        Write a function that returns "A" or "B" based on the current state of the game.

        A full game is always 100 rounds.

        Good luck with this game of social deduction!
        """
        return str(np.random.choice(["A", "B"]))

    # def sorry(self, board: pd.DataFrame, info: dict, dice_roll: int) -> int:
    #     """
    #     This is a slightly modified version of the classic game of Sorry! (Mens erger je niet!)

    #     Every turn you will get a pandas dataframe with the current state of the board. Each row represents a cell on the board.
    #     The columns are:
    #     - "index" which represents the position on the board, starting at 0 and going up to the number of spaces on the board - 1
    #     - "home" which will hold None or the name of the player whose home it is (e.g. "Mees" or "Ivo")
    #     - "space" which will hold either None (empty space) or a piece. Pieces are named as "playername_piecenum" (e.g. "Ivo_1", "Jan_2")

    #     Each turn you will also receive a dictionary with some info on the game, which will have the following keys:
    #     - "pieces_at_home": a dictionary with player names as keys and a list of piece numbers at home as values
    #     - "pieces_finished": a dictionary with player names as keys and a list of piece numbers that have finished as values

    #     Additionally you will receive your current dice roll (a random number in the range [1, 6])
    #     - if this number is a 6, you can choose to move a piece from home to the starting position
    #     - additionally, you can always choose to move a piece on the board, if you have any pieces on the board

    #     You must return the number of the piece you want to move (1-4). This piece will then be moved according to the rules of the game.

    #     Pieces move as follows:
    #     - If you move a piece from home to the starting position, it will be placed on the first space of the board after your home space
    #     - If you move a piece on the board, it will move forward a number of spaces equal to the dice roll
    #         - This means that if you overshoot your home, you will just continue moving!
    #     - If a piece lands on a space occupied by another piece, the other piece is sent back to its home

    #     If you perform an illegal move, a random piece of yours is removed from the board and sent back to your home.
    #     A move is illegal if it refers to a piece that cannot be moved or placed on the board. Returning an invalid value is also illegal.
    #     If no pieces can be removed from the board, nothing happens

    #     SOME EXAMPLE TURNS (with only 2 players, Mark and Dirk):
    #     | index | home  | space |
    #     | 0     | Mark  | None  |
    #     | 1     | None  | None  |
    #     | 2     | None  | None  |
    #     | 3     | None  | None  |
    #     | 4     | None  | None  |
    #     | 5     | Dirk  | None  |
    #     | 6     | None  | None  |
    #     | 7     | None  | None  |
    #     | 8     | None  | None  |
    #     | 9     | None  | None  |
    #     info dict:
    #     {
    #         "pieces_at_home": {"Mark": [1, 2, 3, 4], "Dirk": [1, 2, 3, 4]},
    #         "pieces_finished": {"Mark": [], "Dirk": []},
    #     }

    #     Mark now rolls a 6 and decides to move piece 1 from home to the starting position. The board now looks like this:
    #     | index | home  | space |
    #     | 0     | Mark  | None  |
    #     | 1     | None  | Mark_1|
    #     | 2     | None  | None  |
    #     | 3     | None  | None  |
    #     | 4     | None  | None  |
    #     | 5     | Dirk  | None  |
    #     | 6     | None  | None  |
    #     | 7     | None  | None  |
    #     | 8     | None  | None  |
    #     | 9     | None  | None  |
    #     info dict:
    #     {
    #         "pieces_at_home": {"Mark": [2, 3, 4], "Dirk": [1, 2, 3, 4]},
    #         "pieces_finished": {"Mark": [], "Dirk": []},
    #     }

    #     Several turn later the board looks like this:
    #     | index | home  | space |
    #     | 0     | Mark  | None  |
    #     | 1     | None  | None  |
    #     | 2     | None  | None  |
    #     | 3     | None  | None  |
    #     | 4     | None  | None  |
    #     | 5     | Dirk  | Mark_1|
    #     | 6     | None  | None  |
    #     | 7     | None  | None  |
    #     | 8     | None  | Dirk_1|
    #     | 9     | None  | None  |
    #     info dict:
    #     {
    #         "pieces_at_home": {"Mark": [2, 3, 4], "Dirk": [2, 3, 4]},
    #         "pieces_finished": {"Mark": [], "Dirk": []},
    #     }

    #     Mark now rolls a 3 and decides to move piece 1. The board now looks like this:
    #     | index | home  | space |
    #     | 0     | Mark  | None  |
    #     | 1     | None  | None  |
    #     | 2     | None  | None  |
    #     | 3     | None  | None  |
    #     | 4     | None  | None  |
    #     | 5     | Dirk  | None  |
    #     | 6     | None  | None  |
    #     | 7     | None  | None  |
    #     | 8     | None  | Mark_1|
    #     | 9     | None  | None  |
    #     info dict:
    #     {
    #         "pieces_at_home": {"Mark": [2, 3, 4], "Dirk": [1, 2, 3, 4]},
    #         "pieces_finished": {"Mark": [], "Dirk": []},
    #     }

    #     Dirks piece is sent back to his home.
    #     Dirk now rolls a 3, but since he has no pieces on the board, he cannot move any piece and thus cannot return a valid move. So nothing happens

    #     If mark now rolls a 2, he can move piece 1, by returning the integer 1, to the home space and finish it.
    #     If mark now rolls a 4, he can move piece 1, by returning the integer 1, overshooting his home. The piece will end up on index 3
    #     If mark now rolls a 6, he can either:
    #     - return the integer 1 to move piece 1 past his home, ending up on index 5
    #     - return integer 2, 3 or 4 to move a new piece from home to the starting position, ending up on index 1

    #     Say that mark rolls a 2 and decides to move piece 1, then the board will look like this:
    #     | index | home  | space |
    #     | 0     | Mark  | None  |
    #     | 1     | None  | None  |
    #     | 2     | None  | None  |
    #     | 3     | None  | None  |
    #     | 4     | None  | None  |
    #     | 5     | Dirk  | None  |
    #     | 6     | None  | None  |
    #     | 7     | None  | None  |
    #     | 8     | None  | None  |
    #     | 9     | None  | None  |
    #     info dict:
    #     {
    #         "pieces_at_home": {"Mark": [2, 3, 4], "Dirk": [1, 2, 3, 4]},
    #         "pieces_finished": {"Mark": [1], "Dirk": []},
    #     }

    #     After every full round of turns the game checks if any player has won by getting all 4 pieces finished.
    #     This means two players can tie and both win.

    #     Good luck removing the pieces of your opponents!
    #     """
    #     home_idx = board[board["home"] == self.name].index[0]
    #     board_length = len(board)
    #     at_home = info["pieces_at_home"][self.name]
    #     finished = info["pieces_finished"][self.name]
    #     start_pos = (home_idx + 1) % board_length

    #     my_on_board = {
    #         int(str(board.loc[i, "space"]).split("_")[-1]): i
    #         for i in board.index
    #         if board.loc[i, "space"] is not None
    #         and str(board.loc[i, "space"]).startswith(self.name + "_")
    #     }

    #     opp_home_map = {
    #         p: board[board["home"] == p].index[0]
    #         for p in info["pieces_at_home"].keys()
    #         if p != self.name
    #     }

    #     def dist_to_home(player_home_idx, pos):
    #         return (player_home_idx - pos) % board_length

    #     def is_own(idx):
    #         sp = board.loc[idx, "space"]
    #         return sp is not None and str(sp).startswith(self.name + "_")

    #     def build_opp_positions(exclude_piece=None):
    #         out = {p: [] for p in opp_home_map}
    #         for i in board.index:
    #             sp = board.loc[i, "space"]
    #             if sp is None:
    #                 continue
    #             piece_name = str(sp)
    #             if exclude_piece is not None and piece_name == exclude_piece:
    #                 continue
    #             owner = "_".join(piece_name.split("_")[:-1])
    #             if owner in out:
    #                 out[owner].append(i)
    #         return out

    #     def capture_risk_probability(target_idx, opp_positions):
    #         not_captured = 1.0
    #         for positions in opp_positions.values():
    #             rolls_that_hit = sum(
    #                 1 for pos in positions
    #                 if 1 <= (target_idx - pos) % board_length <= 6
    #             )
    #             p_cap = min(1.0, rolls_that_hit / 6.0)
    #             not_captured *= 1.0 - p_cap
    #         return 1.0 - not_captured

    #     FINISH_SCORE = 100_000
    #     my_finished_count = len(finished)
    #     candidates = []

    #     for pn, pos in my_on_board.items():
    #         new_pos = (pos + dice_roll) % board_length
    #         if is_own(new_pos):
    #             continue  # would send own piece home — skip

    #         score = 0.0
    #         target_sp = board.loc[new_pos, "space"]
    #         target_home = board.loc[new_pos, "home"]
    #         removed_piece = None

    #         if target_home == self.name:
    #             score += FINISH_SCORE
    #             if my_finished_count == 3:
    #                 score += 35_000.0
    #             candidates.append((pn, score))
    #             continue

    #         # Progress toward home
    #         before = dist_to_home(home_idx, pos)
    #         after = dist_to_home(home_idx, new_pos)
    #         score += 50.0 * (before - after)

    #         # Capture bonus
    #         if target_sp is not None and not str(target_sp).startswith(self.name + "_"):
    #             piece_name = str(target_sp)
    #             removed_piece = piece_name
    #             owner = "_".join(piece_name.split("_")[:-1])
    #             opp_home = opp_home_map.get(owner, 0)
    #             opp_dist = dist_to_home(opp_home, new_pos)
    #             opp_fin = len(info["pieces_finished"].get(owner, []))
    #             score += 1500.0 + 60.0 * (board_length - opp_dist) + 330.0 * opp_fin
    #             if opp_dist <= 6:
    #                 score += 900.0
    #             if opp_fin >= 3:
    #                 score += 1500.0

    #         # Risk at target position after move
    #         opp_after = build_opp_positions(exclude_piece=removed_piece)
    #         risk_after = capture_risk_probability(new_pos, opp_after)
    #         my_value = board_length - dist_to_home(home_idx, new_pos)
    #         score -= risk_after * (260.0 + 42.0 * my_value)

    #         # Bonus for escaping current danger
    #         opp_now = build_opp_positions()
    #         risk_now = capture_risk_probability(pos, opp_now)
    #         score += 170.0 * (risk_now - risk_after)

    #         score += float(np.random.uniform(0.0, 1e-3))
    #         candidates.append((pn, score))

    #     if dice_roll == 6 and at_home:
    #         pn = at_home[0]
    #         if not is_own(start_pos):
    #             score = 260.0 if len(my_on_board) < 2 else 110.0
    #             target_sp = board.loc[start_pos, "space"]
    #             removed_piece = None
    #             if target_sp is not None and not str(target_sp).startswith(self.name + "_"):
    #                 piece_name = str(target_sp)
    #                 removed_piece = piece_name
    #                 owner = "_".join(piece_name.split("_")[:-1])
    #                 opp_home = opp_home_map.get(owner, 0)
    #                 opp_dist = dist_to_home(opp_home, start_pos)
    #                 opp_fin = len(info["pieces_finished"].get(owner, []))
    #                 score += 1500.0 + 60.0 * (board_length - opp_dist) + 330.0 * opp_fin
    #                 if opp_dist <= 6:
    #                     score += 900.0
    #                 if opp_fin >= 3:
    #                     score += 1500.0
    #             opp_after = build_opp_positions(exclude_piece=removed_piece)
    #             risk_after = capture_risk_probability(start_pos, opp_after)
    #             my_value = board_length - dist_to_home(home_idx, start_pos)
    #             score -= risk_after * (260.0 + 42.0 * my_value)
    #             score += float(np.random.uniform(0.0, 1e-3))
    #             candidates.append((pn, score))

    #     if candidates:
    #         return max(candidates, key=lambda x: x[1])[0]

    #     for pn in range(1, 5):
    #         if pn not in finished:
    #             if pn in my_on_board or (dice_roll == 6 and pn in at_home):
    #                 return pn
    #     return 1

    def rps_gun(
        self, history: pd.DataFrame
    ) -> tuple[Literal["r", "p", "s", "g", "d"], int]:
        """
        Welcome to ROCK PAPER SCISSORS GUN DUCK!

        The rules are simple:
        - ROCK beats SCISSORS
        - SCISSORS beats PAPER
        - PAPER beats ROCK

        These you know from normal ROCK PAPER SCISSORS, but now there are two new options:
        - GUN has you shoot your gun at the opponent. If you do so, you need to wait 5 rounds until the gun is reloaded and you can use GUN again.
            - GUN beats PAPER and SCISSORS. These options have no recourse against a gun.
            - If your opponent plays ROCK, they have a 50/50 chance to block the bullet.
              If so, they then throw their rock at you, which beats your now bullet-less gun. So it's a 50/50!
            - GUN against GUN is also a 50/50, since you both shoot at the same time, and thus have a 50/50 chance to hit each other.
            - DUCK ducks under the bullet. The ducker then has the opportunity for an easy under-the-belt strike under the table. So DUCK beats GUN.

        - DUCK is the other new option. It has you duck down under the table. You can use it every round, there are no limitations on it.
            - DUCK beats GUN, as explained above.
            - DUCK also beats ROCK, since you can duck under a thrown rock. Again leading to an easy under-the-belt strike.
            - PAPER beats DUCK, since after ducking, the ducker is now rounder, like a rock, and thus vulnerable to paper.
            - SCISSORS beats DUCK, since after ducking, the ducker is now vulnerable to a stab in the exposed neck with scissors.

        The winner of each round receives 100 points. Most points at the end of the game wins!

        In addition to returning your choice ("r" for ROCK, "p" for PAPER, "s" for SCISSORS, "g" for GUN, and "d" for DUCK),
        you can also decide to wager some of your gained points on the round.

        If you return a bet, and you win the round, your bet amount is added to your score.
        If you return a bet, and you lose the round, your bet amount is subtracted from your score.

        You can lend points from the bank by betting more points than you currently have. You can lend up to 2000 points per round.
        If you bet more points than you currently have, and then lose the round, you will end up with a debt. This debt has 10% interest per round.
        EXAMPLE:
            - ROUND 0
            - You start with 0 points.
            - You win the round and gain 100 points.
            - You now have 100 points.
            - ROUND 1
            - You have 100 points.
            - You bet 300 points on a round, and lose. You now have -200 points.
            - ROUND 2
            - Your debt increases by 10% interest. You now have -220 points.
            - Play continues as normal

        You receive the complete history of the game in a pandas dataframe, with a row for each played round, and the following columns:
        - `your name`: Your choice in this round ("r", "p", "s", "g", or "d")
        - `opponents name`: the choice of your opponent in this round ("r", "p", "s", "g", or "d")
        - `your name`_score: Your score after this round (int)
        - `opponents name`_score: the score of your opponent after this round (int)
        - `your name`_reload_timer: the number of rounds until your gun is reloaded and you can use GUN again
                                    (so if this is 1 at the LAST round in the table you can use GUN in THIS round)
        - `opponents name`_reload_timer: the number of rounds until your opponent's gun is reloaded and they can use GUN again
                                         (so if this is 1 at the LAST round in the table your opponent can use GUN in THIS round)
        - rounds_remaining: the number of rounds remaining in the game (int)

        Example input:

          dummy Opponent dummy_score Opponent_score dummy_reload_timer Opponent_reload_timer rounds_remaining  dummy_bet  Opponent_bet
        0     g        s        -100            200                  4                     4                4      100.0         100.0
        1     s        p          90            100                  3                     3                3      100.0         100.0
        2     s        p         290              0                  2                     2                2      100.0         100.0
        3     g        r         190            200                  1                     1                1      100.0         100.0
        4     g        g         390            100                  0                     0                0      100.0         100.0

        A full game is always 400 rounds.
        If you return an invalid move (e.g. GUN when not reloaded, lending more than the allowed 2000 per round), you forfeit the round and your opponent wins by default.

        Good luck with this extremely logical and strategic game of ROCK PAPER SCISSORS GUN DUCK!
        """
        MOVES = ["r", "p", "s", "g", "d"]
        WIN = {
            ("r", "r"): 0,
            ("r", "p"): -1,
            ("r", "s"): 1,
            ("r", "g"): 0,
            ("r", "d"): -1,
            ("p", "r"): 1,
            ("p", "p"): 0,
            ("p", "s"): -1,
            ("p", "g"): -1,
            ("p", "d"): 1,
            ("s", "r"): -1,
            ("s", "p"): 1,
            ("s", "s"): 0,
            ("s", "g"): -1,
            ("s", "d"): 1,
            ("g", "r"): 0,
            ("g", "p"): 1,
            ("g", "s"): 1,
            ("g", "g"): 0,
            ("g", "d"): -1,
            ("d", "r"): 1,
            ("d", "p"): -1,
            ("d", "s"): -1,
            ("d", "g"): 1,
            ("d", "d"): 0,
        }

        if len(history) == 0:
            return ("p", 0)

        # Resolve column names
        timer_cols = [c for c in history.columns if c.endswith("_reload_timer")]
        my_timer_col = self.name + "_reload_timer"
        opp_timer_col = next(c for c in timer_cols if c != my_timer_col)
        opp_name = opp_timer_col[: -len("_reload_timer")]

        last = history.iloc[-1]
        my_last_timer = int(last[my_timer_col])
        my_last_choice = str(last[self.name])
        opp_last_timer = int(last[opp_timer_col])
        opp_last_choice = str(last[opp_name])

        can_gun = (my_last_timer == 1) or (my_last_timer == 0 and my_last_choice != "g")
        opp_can_gun = (opp_last_timer == 1) or (
            opp_last_timer == 0 and opp_last_choice != "g"
        )
        my_score = int(last[self.name + "_score"])
        rounds_remaining = int(last["rounds_remaining"])

        # Init or reset state on new game
        if self._rps_state is None or len(history) < self._rps_state["round"]:
            self._rps_state = {
                "round": 0,
                "op_last": None,
                "op_prev": None,
                "my_last": None,
                "global_counts": {},
                "phase_counts": {},
                "after_op_counts": {},
                "after_my_counts": {},
                "bigram_counts": {},
            }

        st = self._rps_state
        opp_choices = history[opp_name].tolist()
        my_choices = history[self.name].tolist()

        # Incrementally update all 5 signals from unprocessed history rows
        while st["round"] < len(history):
            i = st["round"]
            last_op = opp_choices[i]
            last_my = my_choices[i]
            phase = i % 6

            st["global_counts"][last_op] = st["global_counts"].get(last_op, 0) + 1
            if phase not in st["phase_counts"]:
                st["phase_counts"][phase] = {}
            st["phase_counts"][phase][last_op] = (
                st["phase_counts"][phase].get(last_op, 0) + 1
            )

            if st["op_last"] is not None:
                aoc = st["after_op_counts"].setdefault(st["op_last"], {})
                aoc[last_op] = aoc.get(last_op, 0) + 1

            if st["my_last"] is not None:
                amc = st["after_my_counts"].setdefault(st["my_last"], {})
                amc[last_op] = amc.get(last_op, 0) + 1

            if st["op_prev"] is not None and st["op_last"] is not None:
                bgc = st["bigram_counts"].setdefault((st["op_prev"], st["op_last"]), {})
                bgc[last_op] = bgc.get(last_op, 0) + 1

            st["op_prev"] = st["op_last"]
            st["op_last"] = last_op
            st["my_last"] = last_my
            st["round"] += 1

        legal_moves = ["r", "p", "s", "d"]
        if can_gun:
            legal_moves.append("g")

        def probs(counts: dict, alpha: float = 1.0) -> dict:
            total = sum(counts.get(m, 0) for m in MOVES) + alpha * len(MOVES)
            return {m: (counts.get(m, 0) + alpha) / total for m in MOVES}

        n = st["round"]
        w_global = 0.35
        w_phase = 0.15
        w_after_op = min(0.30, 0.08 + n / 800)
        w_after_my = min(0.20, 0.05 + n / 1000)

        bigram_key = (
            (st["op_prev"], st["op_last"]) if st["op_prev"] is not None else None
        )
        bigram_data = st["bigram_counts"].get(bigram_key, {}) if bigram_key else {}
        bigram_n = sum(bigram_data.values())
        w_bigram = min(0.15, bigram_n / 80) if bigram_n >= 5 else 0.0

        w_sum = w_global + w_phase + w_after_op + w_after_my + w_bigram
        p_global = probs(st["global_counts"], alpha=2.0)
        p_phase = probs(st["phase_counts"].get(n % 6, {}), alpha=1.0)
        p_after_op = probs(st["after_op_counts"].get(st["op_last"], {}), alpha=1.0)
        p_after_my = probs(st["after_my_counts"].get(st["my_last"], {}), alpha=1.0)
        p_bigram = probs(bigram_data, alpha=1.0)

        op_dist = {
            m: (
                w_global * p_global[m]
                + w_phase * p_phase[m]
                + w_after_op * p_after_op[m]
                + w_after_my * p_after_my[m]
                + w_bigram * p_bigram[m]
            )
            / w_sum
            for m in MOVES
        }

        # Zero out gun probability when opponent can't fire
        if not opp_can_gun:
            non_g = sum(op_dist[m] for m in ["r", "p", "s", "d"])
            if non_g > 0:
                for m in ["r", "p", "s", "d"]:
                    op_dist[m] /= non_g
            op_dist["g"] = 0.0

        # Compute expected utility per legal move
        utils = {
            m: sum(WIN.get((m, op), 0) * p for op, p in op_dist.items())
            for m in legal_moves
        }

        # Softmax action selection — less exploitable than pure argmax
        tau = max(0.06, 0.35 - n / 600)
        max_u = max(utils.values())
        weights = {m: float(np.exp((utils[m] - max_u) / tau)) for m in legal_moves}
        z = sum(weights.values())
        policy = {m: weights[m] / z for m in legal_moves}
        chosen = str(np.random.choice(legal_moves, p=[policy[m] for m in legal_moves]))

        # Edge and entropy for bet sizing
        p_win = p_lose = 0.0
        for op, p in op_dist.items():
            wv = WIN.get((chosen, op), 0)
            if wv == 1:
                p_win += p
            elif wv == -1:
                p_lose += p
            else:
                p_win += 0.5 * p
                p_lose += 0.5 * p

        edge = p_win - p_lose
        entropy = -sum(p * np.log(p + 1e-12) for p in op_dist.values()) / np.log(5)
        confidence = float(np.clip(1.0 - entropy, 0.0, 1.0))

        max_legal_bet = max(0, int(my_score)) + 2000
        if edge <= 0:
            bet = 0
        else:
            urgency = 1.5 if rounds_remaining < 60 else 1.0
            if rounds_remaining < 20:
                urgency = 2.2
            raw_bet = max_legal_bet * edge * (0.15 + 0.85 * confidence) * 0.65 * urgency
            bet = int(np.clip(raw_bet, 0, max_legal_bet))

        st["my_last"] = chosen
        return (cast(Literal["r", "p", "s", "g", "d"], chosen), bet)
