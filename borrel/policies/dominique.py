import random
from collections import deque
from typing import Literal

import numpy as np
import pandas as pd


class Dominique:
    def __init__(self, name: str = "dominique"):
        # NOTE: DO NOT TOUCH!
        self.name = name  # NOTE: DO NOT TOUCH!
        # NOTE: DO NOT TOUCH!

        # Feel free to store whatever you want here.
        # Each full run of a game will use a fresh instance of this class.
        # So for for example battleship, a new instance will be created for each game, but not for each turn.

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

        ### Stolen from Carlos last year, thanks Carlito! ###

        self.tron_grid = grid
        self.tron_height = len(grid)
        self.tron_width = len(grid[0])

        self.tron_dirs = {
            "up": (-1, 0),
            "down": (1, 0),
            "left": (0, -1),
            "right": (0, 1),
        }

        dir_symbols = {">": "right", "<": "left", "^": "up", "v": "down"}

        for r in range(self.tron_height):
            for c in range(self.tron_width):
                ch = grid[r][c]
                if ch in dir_symbols:
                    self_pos = (r, c)
                    self_dir = dir_symbols[ch]

        opp_pos = None
        for r in range(self.tron_height):
            for c in range(self.tron_width):
                if grid[r][c] == "X":
                    opp_pos = (r, c)

        legal_moves = []
        for move, (dr, dc) in self.tron_dirs.items():
            nr, nc = self_pos[0] + dr, self_pos[1] + dc
            if self._tron_is_safe(nr, nc):
                legal_moves.append(move)

        if not legal_moves:
            return self_dir

        best_move = None
        best_score = -1e18
        for move in legal_moves:
            nr = self_pos[0] + self.tron_dirs[move][0]
            nc = self_pos[1] + self.tron_dirs[move][1]
            my_area = self._tron_flood_fill_area(nr, nc)

            # Counter-map awareness: avoid trajectories that hand the opponent
            # clearly larger free space after this exchange.
            opp_best_area = 0
            head_distance = 0
            if opp_pos is not None:
                head_distance = abs(nr - opp_pos[0]) + abs(nc - opp_pos[1])
                for odr, odc in self.tron_dirs.values():
                    orow, ocol = opp_pos[0] + odr, opp_pos[1] + odc
                    if self._tron_is_safe(orow, ocol):
                        opp_best_area = max(
                            opp_best_area, self._tron_flood_fill_area(orow, ocol)
                        )

            score = (
                1.0 * my_area
                - 0.35 * opp_best_area
                + 0.45 * head_distance
                + float(np.random.uniform(0.0, 1e-3))
            )
            if score > best_score:
                best_score = score
                best_move = move

        return best_move

    def _tron_is_safe(self, r: int, c: int) -> bool:
        if not (0 <= r < self.tron_height and 0 <= c < self.tron_width):
            return False
        return self.tron_grid[r][c] == " "

    def _tron_flood_fill_area(self, r: int, c: int) -> int:
        if not self._tron_is_safe(r, c):
            return 0

        visited = set()
        queue = deque([(r, c)])
        visited.add((r, c))
        area = 0

        while queue:
            cr, cc = queue.popleft()
            area += 1
            for dr, dc in self.tron_dirs.values():
                nr, nc = cr + dr, cc + dc
                if (nr, nc) not in visited and self._tron_is_safe(nr, nc):
                    visited.add((nr, nc))
                    queue.append((nr, nc))

        return area

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

        def legal_moves(h: np.ndarray, v: np.ndarray) -> list[dict]:
            moves = []
            for r in range(size + 1):
                for c in range(size):
                    if not h[r, c]:
                        moves.append({"orientation": "h", "row": r, "col": c})
            for r in range(size):
                for c in range(size + 1):
                    if not v[r, c]:
                        moves.append({"orientation": "v", "row": r, "col": c})
            return moves

        def box_sides(h: np.ndarray, v: np.ndarray, r: int, c: int) -> int:
            return int(h[r, c]) + int(h[r + 1, c]) + int(v[r, c]) + int(v[r, c + 1])

        def affected_boxes(move: dict) -> list[tuple[int, int]]:
            r = move["row"]
            c = move["col"]
            if move["orientation"] == "h":
                out = []
                if r > 0:
                    out.append((r - 1, c))
                if r < size:
                    out.append((r, c))
                return out
            out = []
            if c > 0:
                out.append((r, c - 1))
            if c < size:
                out.append((r, c))
            return out

        def apply_move(
            h: np.ndarray, v: np.ndarray, move: dict
        ) -> tuple[np.ndarray, np.ndarray]:
            nh = h.copy()
            nv = v.copy()
            if move["orientation"] == "h":
                nh[move["row"], move["col"]] = True
            else:
                nv[move["row"], move["col"]] = True
            return nh, nv

        def immediate_boxes_closed(h: np.ndarray, v: np.ndarray, move: dict) -> int:
            nh, nv = apply_move(h, v, move)
            closed = 0
            for br, bc in affected_boxes(move):
                if box_sides(nh, nv, br, bc) == 4:
                    closed += 1
            return closed

        def gives_third_side(h: np.ndarray, v: np.ndarray, move: dict) -> int:
            nh, nv = apply_move(h, v, move)
            count = 0
            for br, bc in affected_boxes(move):
                if box_sides(nh, nv, br, bc) == 3:
                    count += 1
            return count

        def opponent_best_take(h: np.ndarray, v: np.ndarray) -> int:
            # Approximate "normal" opponent: greedily take immediate boxes.
            best = 0
            for mv in legal_moves(h, v):
                best = max(best, immediate_boxes_closed(h, v, mv))
            return best

        moves = legal_moves(horizontal_lines, vertical_lines)
        lines_remaining = int(np.size(horizontal_lines) + np.size(vertical_lines)) - int(
            np.sum(horizontal_lines) + np.sum(vertical_lines)
        )
        endgame_pressure = max(0.0, (20.0 - float(lines_remaining)) / 20.0)
        if not moves:
            return {"orientation": "h", "row": 0, "col": 0}

        # Stage 1: baseline strong play -> always take points now.
        scoring = [
            m
            for m in moves
            if immediate_boxes_closed(horizontal_lines, vertical_lines, m) > 0
        ]
        if scoring:
            best_move = None
            best_score = -1e18
            for m in scoring:
                now = immediate_boxes_closed(horizontal_lines, vertical_lines, m)
                nh, nv = apply_move(horizontal_lines, vertical_lines, m)
                # Stage 2 anti-baseline: among scoring moves, avoid handing over easy counter-scoring.
                opp_take = opponent_best_take(nh, nv)
                thirds = gives_third_side(horizontal_lines, vertical_lines, m)
                score = (
                    500.0 * now
                    - (180.0 + 100.0 * endgame_pressure) * opp_take
                    - (40.0 + 15.0 * endgame_pressure) * thirds
                    + float(np.random.uniform(0, 1e-3))
                )
                if score > best_score:
                    best_score = score
                    best_move = m
            return best_move

        # No immediate score: try to play "safe" moves that create zero 3-sided boxes.
        safe = [
            m
            for m in moves
            if gives_third_side(horizontal_lines, vertical_lines, m) == 0
        ]
        candidate_moves = safe if safe else moves

        # Anti-baseline layer: choose move that minimizes opponent's immediate gain next turn,
        # then secondarily limits creation of 3-sided boxes and preserves flexibility.
        best_move = None
        best_score = -1e18
        for m in candidate_moves:
            nh, nv = apply_move(horizontal_lines, vertical_lines, m)
            opp_take = opponent_best_take(nh, nv)
            thirds = gives_third_side(horizontal_lines, vertical_lines, m)
            future_safe = sum(
                1
                for mv2 in legal_moves(nh, nv)
                if gives_third_side(nh, nv, mv2) == 0
            )

            # Prefer central edges early; many standard bots overvalue this,
            # but only after tactical safety checks.
            if m["orientation"] == "h":
                center_dist = abs(m["row"] - size / 2) + abs(m["col"] - (size - 1) / 2)
            else:
                center_dist = abs(m["row"] - (size - 1) / 2) + abs(m["col"] - size / 2)

            score = (
                -(260.0 + 140.0 * endgame_pressure) * opp_take
                - (70.0 + 40.0 * endgame_pressure) * thirds
                + 1.5 * future_safe
                - 2.0 * center_dist
                + float(np.random.uniform(0, 1e-3))
            )
            if score > best_score:
                best_score = score
                best_move = m

        return best_move

    def sorry(self, board: pd.DataFrame, info: dict, dice_roll: int) -> int:
        """
        This is a slightly modified version of the classic game of Sorry! (Mens erger je niet!)

        Every turn you will get a pandas dataframe with the current state of the board. Each row represents a cell on the board.
        The columns are:
        - "index" which represents the position on the board, starting at 0 and going up to the number of spaces on the board - 1
        - "home" which will hold None or the name of the player whose home it is (e.g. "Mees" or "Ivo")
        - "space" which will hold either None (empty space) or a piece. Pieces are named as "playername_piecenum" (e.g. "Ivo_1", "Jan_2")

        Each turn you will also receive a dictionary with some info on the game, which will have the following keys:
        - "pieces_at_home": a dictionary with player names as keys and a list of piece numbers at home as values
        - "pieces_finished": a dictionary with player names as keys and a list of piece numbers that have finished as values

        Additionally you will receive your current dice roll (a random number in the range [1, 6])
        - if this number is a 6, you can choose to move a piece from home to the starting position
        - additionally, you can always choose to move a piece on the board, if you have any pieces on the board

        You must return the number of the piece you want to move (1-4). This piece will then be moved according to the rules of the game.

        Pieces move as follows:
        - If you move a piece from home to the starting position, it will be placed on the first space of the board after your home space
        - If you move a piece on the board, it will move forward a number of spaces equal to the dice roll
            - This means that if you overshoot your home, you will just continue moving!
        - If a piece lands on a space occupied by another piece, the other piece is sent back to its home

        If you perform an illegal move, a random piece of yours is removed from the board and sent back to your home.
        A move is illegal if it refers to a piece that cannot be moved or placed on the board. Returning an invalid value is also illegal.
        If no pieces can be removed from the board, nothing happens

        SOME EXAMPLE TURNS (with only 2 players, Mark and Dirk):
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
            "pieces_at_home": {"Mark": [2, 3, 4], "Dirk": [2, 3, 4]},
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
        Dirk now rolls a 3, but since he has no pieces on the board, he cannot move any piece and thus cannot return a valid move. So nothing happens

        If mark now rolls a 2, he can move piece 1, by returning the integer 1, to the home space and finish it.
        If mark now rolls a 4, he can move piece 1, by returning the integer 1, overshooting his home. The piece will end up on index 3
        If mark now rolls a 6, he can either:
        - return the integer 1 to move piece 1 past his home, ending up on index 5
        - return integer 2, 3 or 4 to move a new piece from home to the starting position, ending up on index 1

        Say that mark rolls a 2 and decides to move piece 1, then the board will look like this:
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
            "pieces_at_home": {"Mark": [2, 3, 4], "Dirk": [1, 2, 3, 4]},
            "pieces_finished": {"Mark": [1], "Dirk": []},
        }

        After every full round of turns the game checks if any player has won by getting all 4 pieces finished.
        This means two players can tie and both win.

        Good luck removing the pieces of your opponents!
        """
        board_len = len(board)
        my_home = int(board.index[board["home"] == self.name][0])

        home_idx = {
            str(player): int(board.index[board["home"] == player][0])
            for player in info["pieces_at_home"].keys()
        }

        spaces = list(board["space"].values)
        piece_to_idx: dict[str, int] = {}
        for idx, piece in enumerate(spaces):
            if piece is not None:
                piece_to_idx[str(piece)] = idx

        my_on_board = {
            int(piece_name.rsplit("_", 1)[1]): idx
            for piece_name, idx in piece_to_idx.items()
            if piece_name.startswith(f"{self.name}_")
        }
        at_home = set(info["pieces_at_home"][self.name])

        def build_opp_positions(
            exclude_piece: str | None = None,
        ) -> dict[str, list[int]]:
            out: dict[str, list[int]] = {
                p: [] for p in info["pieces_at_home"].keys() if p != self.name
            }
            for idx, piece in enumerate(spaces):
                if piece is None:
                    continue
                piece_name = str(piece)
                if exclude_piece is not None and piece_name == exclude_piece:
                    continue
                owner, _ = piece_name.rsplit("_", 1)
                if owner != self.name:
                    out.setdefault(owner, []).append(idx)
            return out

        def dist_to_home(player: str, idx: int) -> int:
            return (home_idx[player] - idx) % board_len

        def capture_risk_probability(
            target_idx: int, opp_positions: dict[str, list[int]]
        ) -> float:
            not_captured_prob = 1.0
            for positions in opp_positions.values():
                p_capture = 0.0
                for pos in positions:
                    d = (target_idx - pos) % board_len
                    if 1 <= d <= 6:
                        p_capture += 1.0 / 6.0
                    elif 7 <= d <= 10:
                        # Light two-turn pressure signal for better medium-term safety.
                        p_capture += 0.08
                p_capture = min(1.0, p_capture)
                not_captured_prob *= 1.0 - p_capture
            return 1.0 - not_captured_prob

        movable = list(my_on_board.keys())
        if dice_roll == 6:
            movable.extend([p for p in at_home if p not in movable])
        if not movable:
            return 1

        pieces_on_board_count = len(my_on_board)
        my_finished = len(info["pieces_finished"][self.name])
        opp_best_finished = max(
            (
                len(info["pieces_finished"].get(player, []))
                for player in info["pieces_finished"].keys()
                if player != self.name
            ),
            default=0,
        )
        best_piece = movable[0]
        best_score = -1e18
        evaluations: list[dict] = []

        for piece_nr in movable:
            from_home = piece_nr in at_home and piece_nr not in my_on_board
            if from_home:
                current_idx = None
                target_idx = (my_home + 1) % board_len
            else:
                current_idx = my_on_board[piece_nr]
                target_idx = (current_idx + dice_roll) % board_len

            target_home = board.loc[target_idx, "home"]
            target_piece = board.loc[target_idx, "space"]
            target_piece_name = str(target_piece) if target_piece is not None else None
            finishing_move = (not from_home) and (target_home == self.name)

            score = 0.0

            # Winning race acceleration.
            if finishing_move:
                score += 12000.0
                if my_finished == 3:
                    score += 35000.0

            # Piece development and progress.
            if from_home:
                score += 260.0 if pieces_on_board_count < 2 else 110.0
            else:
                before = dist_to_home(self.name, current_idx)
                after = dist_to_home(self.name, target_idx)
                progress = before if finishing_move else (before - after) % board_len
                score += 50.0 * progress

            # Occupant interaction (capture, self-block, or spawn-overwrite quirk).
            removed_piece = None
            if target_piece_name is not None:
                owner, _ = target_piece_name.rsplit("_", 1)
                if owner == self.name:
                    score -= 5000.0
                else:
                    removed_piece = target_piece_name
                    opp_dist = dist_to_home(owner, target_idx)
                    opp_finished = len(info["pieces_finished"].get(owner, []))

                    score += 1500.0
                    score += 60.0 * (board_len - opp_dist)
                    score += 330.0 * opp_finished

                    # Extra value for denying near-finish threats.
                    if opp_dist <= 6:
                        score += 900.0
                    if opp_finished >= 3:
                        score += 1500.0

                    # Engine quirk: entering from home overwrites occupancy directly.
                    if from_home:
                        score += 1200.0

                    # If we are behind in finished pieces, prioritize disruption.
                    if my_finished < opp_best_finished:
                        score += 450.0

            # Risk evaluation after hypothetical move.
            if not finishing_move:
                opp_after = build_opp_positions(exclude_piece=removed_piece)
                risk_after = capture_risk_probability(target_idx, opp_after)

                my_piece_value = board_len - dist_to_home(self.name, target_idx)
                score -= risk_after * (260.0 + 42.0 * my_piece_value)

                # Prefer moves that reduce current danger on an exposed runner.
                if current_idx is not None:
                    opp_now = build_opp_positions(exclude_piece=None)
                    risk_now = capture_risk_probability(current_idx, opp_now)
                    score += 170.0 * (risk_now - risk_after)

                # When ahead in the race, preserve board position more strongly.
                if my_finished > opp_best_finished:
                    score -= 140.0 * risk_after
            else:
                risk_after = 0.0

            # Tiny jitter to prevent deterministic mirrors.
            score += float(np.random.uniform(0.0, 1e-3))

            if score > best_score:
                best_score = score
                best_piece = piece_nr

            evaluations.append(
                {
                    "piece": piece_nr,
                    "score": score,
                    "finishing": finishing_move,
                    "risk": float(risk_after),
                    "from_home": from_home,
                    "dist_after": 0 if finishing_move else dist_to_home(self.name, target_idx),
                }
            )

        tied = [
            ev
            for ev in evaluations
            if abs(float(ev["score"]) - float(best_score)) <= 2.0
        ]
        if len(tied) > 1:
            tied.sort(
                key=lambda ev: (
                    1 if ev["finishing"] else 0,
                    -ev["risk"],
                    0 if ev["from_home"] else 1,
                    -ev["dist_after"],
                    float(np.random.uniform(0.0, 1e-6)),
                ),
                reverse=True,
            )
            best_piece = int(tied[0]["piece"])

        return int(best_piece)

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
        if history.empty or not hasattr(self, "_rps_state"):
            self._rps_state = {
                "round": 0,
                "my_reload": 0,
                "op_reload": 0,
                "my_last": None,
                "op_last": None,
                "phase_counts": {},
                "global_counts": {},
                "after_op_counts": {},
                "after_my_counts": {},
            }

        st = self._rps_state
        moves = ["r", "p", "s", "g", "d"]

        # If a new game starts, reset state.
        if len(history) < st["round"]:
            st.update(
                {
                    "round": 0,
                    "my_reload": 0,
                    "op_reload": 0,
                    "my_last": None,
                    "op_last": None,
                    "phase_counts": {},
                    "global_counts": {},
                    "after_op_counts": {},
                    "after_my_counts": {},
                }
            )

        name_cols = [
            c
            for c in history.columns
            if c != "rounds_remaining"
            and not c.endswith("_score")
            and not c.endswith("_reload_timer")
            and not c.endswith("_bet")
        ]
        opp_name = next((c for c in name_cols if c != self.name), None)
        if opp_name is None:
            # Fallback for malformed history: safe random non-gun opening.
            return (str(np.random.choice(["r", "p", "s", "d"])), 0)

        my_score_col = f"{self.name}_score"
        rounds_remaining = (
            int(history.iloc[-1]["rounds_remaining"]) if not history.empty else 399
        )
        my_score = int(history.iloc[-1][my_score_col]) if not history.empty else 0

        # Decrement our own inferred reload before selecting this round's action.
        if st["round"] < len(history):
            st["my_reload"] = max(0, st["my_reload"] - 1)
            st["op_reload"] = max(0, st["op_reload"] - 1)

            last_my = str(history.iloc[-1][self.name])
            last_op = str(history.iloc[-1][opp_name])

            if last_my == "g":
                st["my_reload"] = 5
            if last_op == "g":
                st["op_reload"] = 5

            st["global_counts"][last_op] = st["global_counts"].get(last_op, 0) + 1

            phase = st["round"] % 6
            if phase not in st["phase_counts"]:
                st["phase_counts"][phase] = {}
            st["phase_counts"][phase][last_op] = (
                st["phase_counts"][phase].get(last_op, 0) + 1
            )

            if st["op_last"] is not None:
                if st["op_last"] not in st["after_op_counts"]:
                    st["after_op_counts"][st["op_last"]] = {}
                st["after_op_counts"][st["op_last"]][last_op] = (
                    st["after_op_counts"][st["op_last"]].get(last_op, 0) + 1
                )

            if st["my_last"] is not None:
                if st["my_last"] not in st["after_my_counts"]:
                    st["after_my_counts"][st["my_last"]] = {}
                st["after_my_counts"][st["my_last"]][last_op] = (
                    st["after_my_counts"][st["my_last"]].get(last_op, 0) + 1
                )

            st["my_last"] = last_my
            st["op_last"] = last_op
            st["round"] = len(history)

        legal_moves = ["r", "p", "s", "d"]
        if st["my_reload"] == 0:
            legal_moves.append("g")

        def probs_from_counts(
            counts: dict[str, int], alpha: float = 1.0
        ) -> dict[str, float]:
            total = sum(counts.get(m, 0) for m in moves) + alpha * len(moves)
            return {m: (counts.get(m, 0) + alpha) / total for m in moves}

        global_p = probs_from_counts(st["global_counts"], alpha=2.0)
        phase_p = probs_from_counts(
            st["phase_counts"].get(st["round"] % 6, {}), alpha=1.0
        )
        after_op_p = probs_from_counts(
            st["after_op_counts"].get(st["op_last"], {}), alpha=1.0
        )
        after_my_p = probs_from_counts(
            st["after_my_counts"].get(st["my_last"], {}), alpha=1.0
        )

        n = max(1, st["round"])
        w_global = 0.35
        w_phase = 0.15
        w_after_op = min(0.35, 0.1 + n / 800)
        w_after_my = min(0.25, 0.05 + n / 1000)
        w_sum = w_global + w_phase + w_after_op + w_after_my
        w_global, w_phase, w_after_op, w_after_my = (
            w_global / w_sum,
            w_phase / w_sum,
            w_after_op / w_sum,
            w_after_my / w_sum,
        )

        op_dist = {
            m: (
                w_global * global_p[m]
                + w_phase * phase_p[m]
                + w_after_op * after_op_p[m]
                + w_after_my * after_my_p[m]
            )
            for m in moves
        }

        # Opponent cannot legally fire while reloading; bias away from impossible gun predictions.
        if st["op_reload"] > 0:
            non_g_total = sum(op_dist[m] for m in ["r", "p", "s", "d"])
            if non_g_total > 0:
                for m in ["r", "p", "s", "d"]:
                    op_dist[m] /= non_g_total
                op_dist["g"] = 0.0

        def round_payoff(my_move: str, op_move: str) -> float:
            if my_move == op_move:
                return 0.0
            if (my_move, op_move) in [("g", "r"), ("r", "g")]:
                return 0.0
            if (my_move, op_move) == ("g", "g"):
                return 0.0

            wins = {
                ("r", "s"),
                ("r", "d"),
                ("p", "r"),
                ("p", "d"),
                ("s", "p"),
                ("s", "d"),
                ("g", "p"),
                ("g", "s"),
                ("d", "r"),
                ("d", "g"),
            }
            return 1.0 if (my_move, op_move) in wins else -1.0

        utilities = {
            m: sum(round_payoff(m, op) * p for op, p in op_dist.items())
            for m in legal_moves
        }

        # Stochastic action selection keeps us less exploitable.
        opponent_entropy = -sum(p * np.log(p + 1e-12) for p in op_dist.values()) / np.log(
            5
        )
        tau_floor = 0.18 if opponent_entropy > 0.75 else 0.12
        tau = max(tau_floor, 0.35 - st["round"] / 1800)

        if "g" in utilities:
            # Suppress predictable panic-shots into likely duck/gun counters.
            utilities["g"] -= 0.35 * op_dist.get("d", 0.0) + 0.20 * op_dist.get("g", 0.0)

        max_u = max(utilities.values())
        weights = {m: np.exp((utilities[m] - max_u) / tau) for m in legal_moves}
        z = sum(weights.values())
        policy = {m: weights[m] / z for m in legal_moves}

        chosen_move = str(
            np.random.choice(legal_moves, p=[policy[m] for m in legal_moves])
        )

        # Estimate edge for bet sizing.
        p_win = 0.0
        p_lose = 0.0
        for op, p in op_dist.items():
            if chosen_move == op:
                continue
            if (chosen_move, op) in [("g", "r"), ("r", "g"), ("g", "g")]:
                p_win += 0.5 * p
                p_lose += 0.5 * p
            elif round_payoff(chosen_move, op) > 0:
                p_win += p
            else:
                p_lose += p

        edge = p_win - p_lose
        entropy = -sum(p * np.log(p + 1e-12) for p in op_dist.values()) / np.log(5)
        confidence = float(np.clip(1.0 - entropy, 0.0, 1.0))

        max_legal_bet = max(0, int(my_score + 2000))
        if edge <= 0:
            bet = 0
        else:
            urgency = 1.0
            if rounds_remaining < 40:
                urgency = 1.35
            if rounds_remaining < 15:
                urgency = 1.7

            # Controlled comeback mode after repeated recent losses.
            if len(history) >= 3:
                recent_payoffs = []
                max_back = min(5, len(history))
                for i in range(1, max_back + 1):
                    my_prev = str(history.iloc[-i][self.name])
                    op_prev = str(history.iloc[-i][opp_name])
                    recent_payoffs.append(round_payoff(my_prev, op_prev))
                loss_streak = sum(1 for p in recent_payoffs if p < 0)
                if loss_streak >= 3:
                    urgency *= 1.35

            raw_bet = max_legal_bet * edge * (0.15 + 0.85 * confidence) * 0.55 * urgency
            bet = int(np.clip(raw_bet, 0, max_legal_bet))

        if chosen_move == "g":
            st["my_reload"] = 5
        st["my_last"] = chosen_move

        return chosen_move, bet
