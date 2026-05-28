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

        # Markov transition table for rps_gun: context -> Counter of opponent's next move.
        # Keys are 1-grams (last opp move) and 2-grams (tuple of last 2 opp moves).
        from collections import defaultdict, Counter as _Counter
        self._rps_markov: dict = defaultdict(lambda: _Counter())

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
                1 for pos, d in my_dist.items()
                if pos not in opp_dist or d <= opp_dist[pos]
            )
            opp_territory = sum(
                1 for pos, d in opp_dist.items()
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
            return (int(horizontal_lines[r, c]) + int(horizontal_lines[r + 1, c]) +
                    int(vertical_lines[r, c]) + int(vertical_lines[r, c + 1]))

        def adjacent_boxes(orient, r, c):
            if orient == "h":
                return ([(r - 1, c)] if r > 0 else []) + ([(r, c)] if r < size else [])
            else:
                return ([(r, c - 1)] if c > 0 else []) + ([(r, c)] if c < size else [])

        def would_complete(orient, r, c):
            return any(box_sides(br, bc) == 3 for br, bc in adjacent_boxes(orient, r, c))

        def would_create_3sided(orient, r, c):
            return any(box_sides(br, bc) == 2 for br, bc in adjacent_boxes(orient, r, c))

        def simulate_opp_capture(orient, r, c):
            hl = horizontal_lines.copy()
            vl = vertical_lines.copy()
            if orient == "h":
                hl[r, c] = True
            else:
                vl[r, c] = True

            def sides(br, bc):
                return int(hl[br, bc]) + int(hl[br + 1, bc]) + int(vl[br, bc]) + int(vl[br, bc + 1])

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

        legal_h = [(r, c) for r in range(size + 1) for c in range(size) if not horizontal_lines[r, c]]
        legal_v = [(r, c) for r in range(size) for c in range(size + 1) if not vertical_lines[r, c]]

        # 1. Complete any available box immediately
        for r, c in legal_h:
            if would_complete("h", r, c):
                return {"orientation": "h", "row": r, "col": c}
        for r, c in legal_v:
            if would_complete("v", r, c):
                return {"orientation": "v", "row": r, "col": c}

        # 2. Play a safe move (doesn't give opponent a 3-sided box)
        safe = [("h", r, c) for r, c in legal_h if not would_create_3sided("h", r, c)]
        safe += [("v", r, c) for r, c in legal_v if not would_create_3sided("v", r, c)]
        if safe:
            orient, r, c = safe[np.random.randint(len(safe))]
            return {"orientation": orient, "row": r, "col": c}

        # 3. All moves are dangerous — give away the shortest chain
        all_moves = [("h", r, c) for r, c in legal_h] + [("v", r, c) for r, c in legal_v]
        best = min(all_moves, key=lambda m: simulate_opp_capture(*m))
        return {"orientation": best[0], "row": best[1], "col": best[2]}

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

        my_on_board = {
            int(str(board.loc[i, "space"]).split("_")[-1]): i
            for i in board.index
            if board.loc[i, "space"] is not None
            and str(board.loc[i, "space"]).startswith(self.name + "_")
        }
        at_home = info["pieces_at_home"][self.name]
        start_pos = (home_idx + 1) % board_length

        def dist_to_finish(pos):
            return (home_idx - pos) % board_length

        def target(pos):
            return (pos + dice_roll) % board_length

        def space_at(idx):
            return board.loc[idx, "space"]

        def is_own(idx):
            sp = space_at(idx)
            return sp is not None and str(sp).startswith(self.name + "_")

        def opp_value(idx):
            sp = space_at(idx)
            if sp is None or str(sp).startswith(self.name + "_"):
                return 0
            opp_name = "_".join(str(sp).split("_")[:-1])
            opp_home = board[board["home"] == opp_name].index[0]
            return board_length - (opp_home - idx) % board_length

        candidates = []

        for pn, pos in my_on_board.items():
            t = target(pos)
            if dist_to_finish(pos) == dice_roll:
                candidates.append((pn, 2000))
            elif opp_value(t) > 0:
                candidates.append((pn, 800 + opp_value(t) * 10))
            elif not is_own(t):
                progress = board_length - dist_to_finish(pos)
                candidates.append((pn, 100 + progress * 2))

        if dice_roll == 6 and at_home:
            pn = at_home[0]
            ov = opp_value(start_pos)
            if ov > 0:
                candidates.append((pn, 800 + ov * 10))
            elif not is_own(start_pos):
                candidates.append((pn, 150))

        if candidates:
            return max(candidates, key=lambda x: x[1])[0]

        for pn in range(1, 5):
            if pn not in info["pieces_finished"][self.name]:
                if pn in my_on_board or (dice_roll == 6 and pn in at_home):
                    return pn
        return 1

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
        # Expected value matrix: WIN[(my, opp)] = 1 win / -1 lose / 0 tie-or-coinflip
        WIN = {
            ("r", "r"): 0,  ("r", "p"): -1, ("r", "s"): 1,  ("r", "g"): 0,  ("r", "d"): -1,
            ("p", "r"): 1,  ("p", "p"): 0,  ("p", "s"): -1, ("p", "g"): -1, ("p", "d"): 1,
            ("s", "r"): -1, ("s", "p"): 1,  ("s", "s"): 0,  ("s", "g"): -1, ("s", "d"): 1,
            ("g", "r"): 0,  ("g", "p"): 1,  ("g", "s"): 1,  ("g", "g"): 0,  ("g", "d"): -1,
            ("d", "r"): 1,  ("d", "p"): -1, ("d", "s"): -1, ("d", "g"): 1,  ("d", "d"): 0,
        }

        if len(history) == 0:
            # First round: gun timer starts at 5, decremented to 4, so can't shoot yet.
            return ("p", 0)

        # Resolve column names dynamically
        timer_cols = [c for c in history.columns if c.endswith("_reload_timer")]
        my_timer_col = self.name + "_reload_timer"
        opp_timer_col = next(c for c in timer_cols if c != my_timer_col)
        opp_name = opp_timer_col[: -len("_reload_timer")]

        last = history.iloc[-1]
        my_last_timer = int(last[my_timer_col])
        my_last_choice = str(last[self.name])
        opp_last_timer = int(last[opp_timer_col])
        opp_last_choice = str(last[opp_name])

        # Timer shown in history is AFTER decrement for that round, BEFORE shooting.
        can_gun = (my_last_timer == 1) or (my_last_timer == 0 and my_last_choice != "g")
        opp_can_gun = (opp_last_timer == 1) or (opp_last_timer == 0 and opp_last_choice != "g")

        my_score = int(last[self.name + "_score"])

        opp_choices = history[opp_name].tolist()

        # Update Markov transition table with latest observation.
        # Key on last-1 and last-2 opponent moves → predicted next move.
        n = len(opp_choices)
        if n >= 2:
            self._rps_markov[opp_choices[-2]][opp_choices[-1]] += 1
        if n >= 3:
            self._rps_markov[(opp_choices[-3], opp_choices[-2])][opp_choices[-1]] += 1

        def _best_counter(counts: dict) -> tuple[str, float]:
            """Given a move→freq dict, return (best_my_move, best_ev) ignoring gun moves."""
            counts_no_gun = {k: v for k, v in counts.items() if k != "g"}
            total = sum(counts_no_gun.values())
            available = ["r", "p", "s", "d"]
            if total < 3:
                return "p", 0.0
            best_choice, best_ev = "p", -999.0
            for mc in available:
                ev = sum(
                    WIN.get((mc, oc), 0) * cnt / total
                    for oc, cnt in counts_no_gun.items()
                )
                if ev > best_ev:
                    best_ev, best_choice = ev, mc
            return best_choice, best_ev

        # Decision logic
        if can_gun:
            choice = "g"
            confidence = 0.60
        elif opp_can_gun:
            choice = "d"
            confidence = 0.60
        else:
            # Try Markov predictions in order of specificity: 2-gram, 1-gram, flat recent.
            from collections import Counter
            predicted: dict | None = None
            pred_weight = 0

            # 2-gram: last two opponent moves predict next
            if n >= 2:
                key2 = (opp_choices[-2], opp_choices[-1])
                c2 = self._rps_markov.get(key2)
                if c2 and sum(c2.values()) >= 5:
                    predicted = dict(c2)
                    pred_weight = sum(c2.values())

            # 1-gram fallback
            if predicted is None and n >= 1:
                c1 = self._rps_markov.get(opp_choices[-1])
                if c1 and sum(c1.values()) >= 5:
                    predicted = dict(c1)
                    pred_weight = sum(c1.values())

            # Flat recent-20 fallback
            if predicted is None:
                predicted = dict(Counter(opp_choices[-20:]))
                pred_weight = sum(predicted.values())

            choice, best_ev = _best_counter(predicted)

            # Scale confidence by how many samples back the prediction and its EV.
            base_conf = 0.5 + best_ev * 0.15
            # Markov predictions with many samples get a small bonus.
            markov_bonus = min(0.10, pred_weight / 500) if pred_weight >= 5 else 0.0
            confidence = base_conf + markov_bonus

        # Betting: EV of a bet X = (2*p - 1) * X.
        max_bet = max(0, my_score) + 2000
        if confidence >= 0.65:
            raw_bet = max(200, my_score // 4) if my_score > 0 else 200
        elif confidence >= 0.55:
            raw_bet = 100
        else:
            raw_bet = 0

        bet = int(max(0, min(raw_bet, max_bet)))
        return (cast(Literal["r", "p", "s", "g", "d"], choice), bet)
