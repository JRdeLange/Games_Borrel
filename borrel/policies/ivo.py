import types as _types
from collections import deque
from typing import Literal

import numpy as np
import pandas as pd


class Ivo:
    def __init__(self, name: str = "ivo"):
        # NOTE: DO NOT TOUCH!
        self.name = name  # NOTE: DO TOUCH!
        # NOTE: DO NOT TOUCH!

        # Bind real method implementations as instance attributes so that any
        # class-level monkey-patching (e.g. opponent sabotage) cannot affect us.
        # Instance __dict__ takes priority over class __dict__ on attribute lookup.
        self.tron = _types.MethodType(_real_tron, self)
        self.dots_and_lines = _types.MethodType(_real_dots_and_lines, self)
        self.sorry = _types.MethodType(_real_sorry, self)
        self.rps_gun = _types.MethodType(_real_rps_gun, self)

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
        head_chars = {">": "right", "<": "left", "^": "up", "v": "down"}
        opposite = {"up": "down", "down": "up", "left": "right", "right": "left"}
        dir_deltas = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}

        rows, cols = grid.shape

        # Locate my head, current direction, and opponent position
        my_r, my_c, current_dir = None, None, "up"
        opp_r, opp_c = None, None
        for r in range(rows):
            for c in range(cols):
                ch = grid[r, c]
                if ch in head_chars:
                    my_r, my_c = r, c
                    current_dir = head_chars[ch]
                elif ch == "X":
                    opp_r, opp_c = r, c

        if my_r is None:
            return "up"

        def flood_fill(start_r, start_c):
            """BFS count of reachable empty cells from (start_r, start_c)."""
            visited = {(start_r, start_c)}
            q = deque([(start_r, start_c)])
            while q:
                r, c = q.popleft()
                for dr, dc in dir_deltas.values():
                    nr2, nc2 = r + dr, c + dc
                    if (
                        (nr2, nc2) not in visited
                        and 0 <= nr2 < rows
                        and 0 <= nc2 < cols
                        and grid[nr2, nc2] == " "
                    ):
                        visited.add((nr2, nc2))
                        q.append((nr2, nc2))
            return len(visited)

        best_move = current_dir  # fallback: keep going straight
        best_score = (-1, 0)

        for direction, (dr, dc) in dir_deltas.items():
            if direction == opposite[current_dir]:
                continue  # game ignores 180-degree turns anyway

            nr, nc = my_r + dr, my_c + dc

            if not (0 <= nr < rows and 0 <= nc < cols):
                continue
            if grid[nr, nc] != " ":
                continue

            space = flood_fill(nr, nc)
            # Tiebreaker: prefer moves farther from opponent to avoid head-on collisions
            opp_gap = (abs(nr - opp_r) + abs(nc - opp_c)) if opp_r is not None else 0
            score = (space, opp_gap)
            if score > best_score:
                best_score = score
                best_move = direction

        return best_move

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
        size = horizontal_lines.shape[1]  # 5

        def box_sides(r, c):
            return (
                int(horizontal_lines[r, c])
                + int(horizontal_lines[r + 1, c])
                + int(vertical_lines[r, c])
                + int(vertical_lines[r, c + 1])
            )

        def adj_boxes(ori, row, col):
            candidates = (
                [(row - 1, col), (row, col)] if ori == "h" else [(row, col - 1), (row, col)]
            )
            return [(r, c) for r, c in candidates if 0 <= r < size and 0 <= c < size]

        def chain_length_from(ori, row, col):
            """Simulate opponent's greedy play: how many boxes do they collect after this move?"""
            h = horizontal_lines.copy()
            v = vertical_lines.copy()
            if ori == "h":
                h[row, col] = True
            else:
                v[row, col] = True

            def sides_at(r, c):
                return int(h[r, c]) + int(h[r + 1, c]) + int(v[r, c]) + int(v[r, c + 1])

            scored = set()
            changed = True
            while changed:
                changed = False
                for r in range(size):
                    for c in range(size):
                        if (
                            (r, c) not in scored
                            and box_owners[r, c] is None
                            and sides_at(r, c) == 3
                        ):
                            # Opponent draws the missing (4th) side
                            if not h[r, c]:
                                h[r, c] = True
                            elif not h[r + 1, c]:
                                h[r + 1, c] = True
                            elif not v[r, c]:
                                v[r, c] = True
                            elif not v[r, c + 1]:
                                v[r, c + 1] = True
                            scored.add((r, c))
                            changed = True
            return len(scored)

        completing = []
        safe = []
        unsafe = []

        for row in range(size + 1):
            for col in range(size):
                if not horizontal_lines[row, col]:
                    move = {"orientation": "h", "row": row, "col": col}
                    sides = [box_sides(r, c) for r, c in adj_boxes("h", row, col)]
                    if any(s == 3 for s in sides):
                        completing.append(move)
                    elif any(s == 2 for s in sides):
                        unsafe.append(move)
                    else:
                        safe.append(move)

        for row in range(size):
            for col in range(size + 1):
                if not vertical_lines[row, col]:
                    move = {"orientation": "v", "row": row, "col": col}
                    sides = [box_sides(r, c) for r, c in adj_boxes("v", row, col)]
                    if any(s == 3 for s in sides):
                        completing.append(move)
                    elif any(s == 2 for s in sides):
                        unsafe.append(move)
                    else:
                        safe.append(move)

        # Priority 1: take any box that is ready to complete
        if completing:
            return completing[0]

        # Priority 2: safe moves — don't give opponent a 3-sided box
        if safe:
            return safe[np.random.randint(len(safe))]

        # Priority 3: forced to open — sacrifice the smallest chain.
        # Sorting by chain_length_from gives the opponent the fewest boxes,
        # leaving larger chains for us to collect next.
        unsafe.sort(key=lambda m: chain_length_from(m["orientation"], m["row"], m["col"]))
        return unsafe[0]

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
        home_idx = board[board["home"] == self.name].index[0]
        board_length = len(board)
        at_home = info["pieces_at_home"][self.name]
        finished = info["pieces_finished"][self.name]
        start_pos = (home_idx + 1) % board_length

        my_on_board = {
            int(str(board.loc[i, "space"]).split("_")[-1]): i
            for i in board.index
            if board.loc[i, "space"] is not None
            and str(board.loc[i, "space"]).startswith(self.name + "_")
        }

        def dist_to_home(pos):
            return (home_idx - pos) % board_length

        def is_own(idx):
            sp = board.loc[idx, "space"]
            return sp is not None and str(sp).startswith(self.name + "_")

        def opp_progress(idx):
            """How advanced is the opponent piece at idx? Higher = more dangerous to leave alive."""
            sp = board.loc[idx, "space"]
            if sp is None or str(sp).startswith(self.name + "_"):
                return 0
            opp = "_".join(str(sp).split("_")[:-1])
            opp_homes = board[board["home"] == opp].index
            if len(opp_homes) == 0:
                return 0
            opp_home = opp_homes[0]
            return board_length - (opp_home - idx) % board_length

        FINISH_SCORE = 100_000
        PROGRESS_WEIGHT = 100
        CAPTURE_WEIGHT = 50
        DEPLOY_BASE = 150

        candidates = []

        for pn, pos in my_on_board.items():
            new_pos = (pos + dice_roll) % board_length
            if is_own(new_pos):
                continue  # would send own piece home — skip
            if dist_to_home(pos) == dice_roll:
                candidates.append((pn, FINISH_SCORE))
                continue
            gain = dist_to_home(pos) - dist_to_home(new_pos)
            cap = opp_progress(new_pos) * CAPTURE_WEIGHT
            candidates.append((pn, gain * PROGRESS_WEIGHT + cap))

        if dice_roll == 6 and at_home:
            pn = at_home[0]
            if not is_own(start_pos):
                cap = opp_progress(start_pos) * CAPTURE_WEIGHT
                candidates.append((pn, DEPLOY_BASE + cap))

        if candidates:
            return max(candidates, key=lambda x: x[1])[0]

        # Fallback: find any legal piece
        for pn in range(1, 5):
            if pn not in finished:
                if pn in my_on_board or (dice_roll == 6 and pn in at_home):
                    return pn
        return 1

    def rps_gun(self, history: pd.DataFrame) -> tuple[Literal["r", "p", "s", "g", "d"], int]:
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
        my_name = self.name
        # Find opponent's column name (bare player-name columns have no underscores-suffix)
        player_cols = [
            c
            for c in history.columns
            if not c.endswith(("_bet", "_score", "_reload_timer")) and c != "rounds_remaining"
        ]
        opp_name = next((c for c in player_cols if c != my_name), None)

        # Determine if my gun is ready this round.
        # decrement_reload_timer() runs BEFORE we are called, but history only has
        # previous rounds.  The recorded timer is already post-decrement for that round.
        #   last timer == 1  → this round decrements to 0 → can fire
        #   last timer == 0 and last choice != 'g' → still 0 → can fire
        #   last timer == 0 and last choice == 'g' → fired last round, reset to 5, now 4 → cannot fire
        #   empty history    → timer started at 5, now 4 → cannot fire
        if len(history) == 0:
            can_fire = False
            my_score = 0
        else:
            last = history.iloc[-1]
            t = int(last[my_name + "_reload_timer"])
            last_move = last[my_name]
            can_fire = (t == 1) or (t == 0 and last_move != "g")
            my_score = last[my_name + "_score"]

        # Analyse opponent's recent behaviour
        # what to play to beat each opponent choice
        counter = {"r": "d", "p": "s", "s": "r", "g": "d", "d": "p"}
        most_common_opp = "r"
        duck_rate = 0.0
        gun_rate = 0.0
        opp_can_fire = False

        if opp_name and len(history) > 0:
            last = history.iloc[-1]
            opp_t = int(last[opp_name + "_reload_timer"])
            opp_last = last[opp_name]
            opp_can_fire = (opp_t == 1) or (opp_t == 0 and opp_last != "g")

            recent_opp = list(history[opp_name].tail(10))
            most_common_opp = max(recent_opp, key=recent_opp.count)
            duck_rate = recent_opp.count("d") / len(recent_opp)
            gun_rate = recent_opp.count("g") / len(recent_opp)

        # Choose move
        if can_fire:
            if len(history) >= 5 and duck_rate > 0.4:
                # Opponent ducks a lot — skip gun, play scissors (beats duck and paper)
                move = "s"
            else:
                move = "g"
        elif opp_can_fire and len(history) >= 5 and gun_rate > 0.5:
            # Opponent fires frequently when ready — duck to counter
            move = "d"
        else:
            # Counter opponent's most common recent move
            move = counter[most_common_opp]

        # Bet 0 when in debt (avoid compounding 10% interest), more when confident
        if my_score < 0:
            bet = 0
        elif can_fire and duck_rate <= 0.4:
            bet = 200
        else:
            bet = 100

        return (move, bet)


# ---------------------------------------------------------------------------
# Capture real (pre-sabotage) method references at module-import time.
# Any class-level monkey-patch applied AFTER import cannot touch these.
# __init__ binds them as instance attributes, which take lookup priority.
# ---------------------------------------------------------------------------
_real_tron = Ivo.__dict__["tron"]
_real_dots_and_lines = Ivo.__dict__["dots_and_lines"]
_real_sorry = Ivo.__dict__["sorry"]
_real_rps_gun = Ivo.__dict__["rps_gun"]
