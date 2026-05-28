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

        def count_future_safe(hl, vl):
            count = 0
            for r2 in range(size + 1):
                for c2 in range(size):
                    if hl[r2, c2]:
                        continue
                    boxes = ([(r2 - 1, c2)] if r2 > 0 else []) + ([(r2, c2)] if r2 < size else [])
                    if not any(
                        int(hl[br, bc]) + int(hl[br + 1, bc]) + int(vl[br, bc]) + int(vl[br, bc + 1]) == 2
                        for br, bc in boxes
                    ):
                        count += 1
            for r2 in range(size):
                for c2 in range(size + 1):
                    if vl[r2, c2]:
                        continue
                    boxes = ([(r2, c2 - 1)] if c2 > 0 else []) + ([(r2, c2)] if c2 < size else [])
                    if not any(
                        int(hl[br, bc]) + int(hl[br + 1, bc]) + int(vl[br, bc]) + int(vl[br, bc + 1]) == 2
                        for br, bc in boxes
                    ):
                        count += 1
            return count

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
            future_safe = count_future_safe(hl2, vl2)
            score = (
                -260.0 * opp_take
                - 70.0 * thirds
                + 1.5 * future_safe
                - 2.0 * center_dist
                + float(np.random.uniform(0, 1e-3))
            )
            if score > best_score:
                best_score = score
                best_move = {"orientation": o, "row": r, "col": c}
        return best_move

    def minority(self, history: pd.DataFrame) -> Literal["A", "B"]:
        """
        Each round all players pick A or B. You score if your choice is the minority.
        With 7 players: minority means ≤3 on your side.
        """
        if len(history) == 0:
            return "A" if np.random.random() < 0.5 else "B"

        all_player_cols = [c for c in history.columns if c != "rounds_remaining"]
        opp_names = [c for c in all_player_cols if c != self.name]

        def predict_p_A(choices: list) -> float:
            n = len(choices)
            if n == 0:
                return 0.5
            # Decaying frequency estimate
            decay = 0.93
            w_sum = w_A = 0.0
            for i, c in enumerate(choices):
                w = decay ** (n - 1 - i)
                w_sum += w
                if c == "A":
                    w_A += w
            p = (w_A + 1.0) / (w_sum + 2.0)
            # 1-gram: blend in P(A | last choice)
            last = choices[-1]
            after1 = [choices[i + 1] for i in range(n - 1) if choices[i] == last]
            if len(after1) >= 4:
                g1 = (after1.count("A") + 0.5) / (len(after1) + 1.0)
                w = min(0.5, len(after1) / 20.0)
                p = (1.0 - w) * p + w * g1
            # 2-gram: blend in P(A | last two choices)
            if n >= 2:
                last2 = (choices[-2], choices[-1])
                after2 = [choices[i + 1] for i in range(n - 2) if (choices[i], choices[i + 1]) == last2]
                if len(after2) >= 3:
                    g2 = (after2.count("A") + 0.5) / (len(after2) + 1.0)
                    w = min(0.35, len(after2) / 15.0)
                    p = (1.0 - w) * p + w * g2
            return float(p)

        predicted_A_opps = sum(predict_p_A(history[opp].tolist()) for opp in opp_names)
        # Pick the side predicted to be minority; noise keeps us unpredictable
        threshold = len(opp_names) / 2.0
        noise = np.random.normal(0.0, 0.4)
        return "A" if predicted_A_opps + noise < threshold else "B"

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

        # Suppress gun when opponent likely to duck or counter-gun
        if "g" in utils:
            utils["g"] -= 0.35 * op_dist.get("d", 0.0) + 0.20 * op_dist.get("g", 0.0)

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
            # Increase urgency on loss streaks
            if len(history) >= 3:
                loss_streak = sum(
                    1 for i in range(1, min(5, len(history)) + 1)
                    if WIN.get((str(history.iloc[-i][self.name]), str(history.iloc[-i][opp_name])), 0) < 0
                )
                if loss_streak >= 3:
                    urgency *= 1.35
            raw_bet = max_legal_bet * edge * (0.15 + 0.85 * confidence) * 0.65 * urgency
            bet = int(np.clip(raw_bet, 0, max_legal_bet))

        st["my_last"] = chosen
        return (cast(Literal["r", "p", "s", "g", "d"], chosen), bet)
