from typing import Literal

import numpy as np
import pandas as pd


class Daniel:
    def __init__(self, name: str = "daniel"):
        # NOTE: DO NOT TOUCH!
        self.name = name  # NOTE: DO NOT TOUCH!
        # NOTE: DO NOT TOUCH!

        # Feel free to store whatever you want here.
        # Each full run of a game will use a fresh instance of this class.
        # So for for example battleship, a new instance will be created for each game, but not for each turn.

        # State for minority game
        self.minority_last_round_idx: int = -1
        self.minority_players: list = []
        self.minority_prev_moves: dict = {}
        self.minority_losses: dict = {}
        self.minority_switches: dict = {}
        self.minority_last_won: dict = {}

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

        head_chars = {">" : (0, 1), "<": (0, -1), "^": (-1, 0), "v": (1, 0)}
        my_pos = None
        my_dir = None
        opp_pos = None
        rows, cols = grid.shape

        for r in range(rows):
            for c in range(cols):
                if grid[r, c] in head_chars:
                    my_pos = (r, c)
                    my_dir = head_chars[grid[r, c]]
                elif grid[r, c] == "X":
                    opp_pos = (r, c)

        if my_pos is None:
            return "right"

        DIRS = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}
        delta_to_dir = {v: k for k, v in DIRS.items()}
        current_dir = delta_to_dir.get(my_dir, "up")
        opposite = {"up": "down", "down": "up", "left": "right", "right": "left"}

        def bfs_dist(start):
            if start is None:
                return {}
            dist = {start: 0}
            q = deque([start])
            while q:
                r, c = q.popleft()
                for dr, dc in DIRS.values():
                    nr, nc = r + dr, c + dc
                    if (0 <= nr < rows and 0 <= nc < cols
                            and (nr, nc) not in dist and grid[nr, nc] == " "):
                        dist[(nr, nc)] = dist[(r, c)] + 1
                        q.append((nr, nc))
            return dist

        opp_dist = bfs_dist(opp_pos)

        best_move = None
        best_score = (-1, -1)

        for move_name, (dr, dc) in DIRS.items():
            if move_name == opposite.get(current_dir, ""):
                continue
            nr, nc = my_pos[0] + dr, my_pos[1] + dc
            if not (0 <= nr < rows and 0 <= nc < cols) or grid[nr, nc] != " ":
                continue

            # BFS from next position to measure reachable territory
            my_from_next = {(nr, nc): 0}
            q2 = deque([(nr, nc)])
            while q2:
                r2, c2 = q2.popleft()
                for ddr, ddc in DIRS.values():
                    nnr, nnc = r2 + ddr, c2 + ddc
                    if (0 <= nnr < rows and 0 <= nnc < cols
                            and (nnr, nnc) not in my_from_next
                            and grid[nnr, nnc] == " "):
                        my_from_next[(nnr, nnc)] = my_from_next[(r2, c2)] + 1
                        q2.append((nnr, nnc))

            total = len(my_from_next)
            # Voronoi: count cells I reach before the opponent
            my_voronoi = sum(
                1 for pos, d in my_from_next.items()
                if d < opp_dist.get(pos, 999)
            )
            score = (my_voronoi, total)
            if score > best_score:
                best_score = score
                best_move = move_name

        if best_move is None:
            for move_name, (dr, dc) in DIRS.items():
                nr, nc = my_pos[0] + dr, my_pos[1] + dc
                if 0 <= nr < rows and 0 <= nc < cols and grid[nr, nc] == " ":
                    return move_name
            return "right"

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
            return (int(horizontal_lines[r, c]) + int(horizontal_lines[r + 1, c])
                    + int(vertical_lines[r, c]) + int(vertical_lines[r, c + 1]))

        def adj_boxes(ori, row, col):
            boxes = []
            if ori == "h":
                if row > 0:    boxes.append((row - 1, col))
                if row < size: boxes.append((row, col))
            else:
                if col > 0:    boxes.append((row, col - 1))
                if col < size: boxes.append((row, col))
            return boxes

        def completes(ori, row, col):
            return any(box_sides(r, c) == 3 for r, c in adj_boxes(ori, row, col))

        def gives_3sided(ori, row, col):
            return any(box_sides(r, c) == 2 for r, c in adj_boxes(ori, row, col))

        def chain_len(ori, row, col):
            """Count boxes in the chain opened when we draw this edge."""
            init = [b for b in adj_boxes(ori, row, col) if box_sides(*b) == 2]
            visited, stack = set(), list(init)
            count = 0
            while stack:
                b = stack.pop()
                if b in visited:
                    continue
                r, c = b
                s = box_sides(r, c)
                if s < 2 or s == 4:
                    continue
                visited.add(b)
                count += 1
                open_sides = []
                if not horizontal_lines[r, c]:       open_sides.append(("h", r, c))
                if not horizontal_lines[r + 1, c]:   open_sides.append(("h", r + 1, c))
                if not vertical_lines[r, c]:         open_sides.append(("v", r, c))
                if not vertical_lines[r, c + 1]:     open_sides.append(("v", r, c + 1))
                for orl, nr, nc in open_sides:
                    for nb in adj_boxes(orl, nr, nc):
                        if nb != b and nb not in visited and 2 <= box_sides(*nb) <= 3:
                            stack.append(nb)
            return count

        completing, safe, unsafe = [], [], []

        for row in range(size + 1):
            for col in range(size):
                if not horizontal_lines[row, col]:
                    m = {"orientation": "h", "row": row, "col": col}
                    if completes("h", row, col):       completing.append(m)
                    elif gives_3sided("h", row, col):  unsafe.append(m)
                    else:                              safe.append(m)

        for row in range(size):
            for col in range(size + 1):
                if not vertical_lines[row, col]:
                    m = {"orientation": "v", "row": row, "col": col}
                    if completes("v", row, col):       completing.append(m)
                    elif gives_3sided("v", row, col):  unsafe.append(m)
                    else:                              safe.append(m)

        if completing:
            return completing[0]
        if safe:
            return safe[0]
        unsafe.sort(key=lambda m: chain_len(m["orientation"], m["row"], m["col"]))
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
        board_size = len(board)
        start_pos = (home_idx + 1) % board_size
        at_home = info["pieces_at_home"][self.name]
        finished = info["pieces_finished"][self.name]

        # Build map of piece_nr -> board position for pieces currently on the board
        on_board = {}
        for pn in range(1, 5):
            if pn in finished or pn in at_home:
                continue
            rows = board[board["space"] == f"{self.name}_{pn}"]
            if len(rows) > 0:
                on_board[pn] = rows.index[0]

        def dist_to_home(pos):
            return (home_idx - pos) % board_size

        def opp_progress(idx):
            """How advanced is the opponent piece at idx — higher means closer to finishing."""
            sp = board.loc[idx, "space"]
            if sp is None or str(sp).startswith(self.name + "_"):
                return 0
            opp = "_".join(str(sp).split("_")[:-1])
            opp_homes = board[board["home"] == opp].index
            if len(opp_homes) == 0:
                return 0
            opp_home = opp_homes[0]
            return board_size - (opp_home - idx) % board_size

        def is_own(idx):
            sp = board.loc[idx, "space"]
            return sp is not None and str(sp).startswith(self.name + "_")

        FINISH_SCORE = 100_000
        CAPTURE_WEIGHT = 60
        PROGRESS_WEIGHT = 100
        DEPLOY_BASE = 150

        # Score every legal candidate move
        candidates = []

        for pn, pos in on_board.items():
            new_pos = (pos + dice_roll) % board_size
            if is_own(new_pos):
                continue  # blocked by own piece
            if new_pos == home_idx:
                candidates.append((pn, FINISH_SCORE))
                continue
            gain = dist_to_home(pos) - dist_to_home(new_pos)
            cap = opp_progress(new_pos) * CAPTURE_WEIGHT
            candidates.append((pn, gain * PROGRESS_WEIGHT + cap))

        # Deploy from home on a 6
        if dice_roll == 6 and at_home:
            pn = at_home[0]
            if not is_own(start_pos):
                cap = opp_progress(start_pos) * CAPTURE_WEIGHT
                # Slightly less eager to deploy when already 3 pieces on board
                deploy_score = (DEPLOY_BASE + cap) if len(on_board) < 3 else (50 + cap)
                candidates.append((pn, deploy_score))

        if candidates:
            return max(candidates, key=lambda x: x[1])[0]

        # Fallback: first movable piece
        for pn in range(1, 5):
            if pn in finished:
                continue
            if pn in on_board:
                return pn
            if pn in at_home and dice_roll == 6:
                return pn

        return 1  # nothing movable; game engine will handle it
    
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
        from collections import Counter

        my_name = self.name
        MOVES = ["r", "p", "s", "g", "d"]

        # Expected-value payoff: WIN[(mine, opp)] = +1 win, 0 tie/coinflip, -1 lose
        WIN = {
            ("r", "r"): 0,  ("r", "p"): -1, ("r", "s"):  1, ("r", "g"):  0, ("r", "d"): -1,
            ("p", "r"): 1,  ("p", "p"):  0, ("p", "s"): -1, ("p", "g"): -1, ("p", "d"):  1,
            ("s", "r"): -1, ("s", "p"):  1, ("s", "s"):  0, ("s", "g"): -1, ("s", "d"):  1,
            ("g", "r"): 0,  ("g", "p"):  1, ("g", "s"):  1, ("g", "g"):  0, ("g", "d"): -1,
            ("d", "r"): 1,  ("d", "p"): -1, ("d", "s"): -1, ("d", "g"):  1, ("d", "d"):  0,
        }

        # Find opponent column name
        player_cols = [
            c for c in history.columns
            if not c.endswith(("_bet", "_score", "_reload_timer"))
            and c != "rounds_remaining"
        ]
        opp_name = next((c for c in player_cols if c != my_name), None)

        # Gun availability, score, rounds remaining
        can_fire = False
        my_score = 0
        opp_can_fire = False
        rounds_remaining = 399

        if len(history) > 0:
            last = history.iloc[-1]
            my_reload = int(last[f"{my_name}_reload_timer"])
            my_last_move = str(last[my_name])
            # timer==1 → decrements to 0 this round → can fire
            # timer==0 and didn't fire last round → still 0 → can fire
            can_fire = (my_reload == 1) or (my_reload == 0 and my_last_move != "g")
            my_score = int(last[f"{my_name}_score"])
            rounds_remaining = int(last["rounds_remaining"])
            if opp_name:
                opp_reload = int(last[f"{opp_name}_reload_timer"])
                opp_last_move = str(last[opp_name])
                opp_can_fire = (opp_reload == 1) or (opp_reload == 0 and opp_last_move != "g")

        # --- Multi-signal opponent distribution ---
        opp_global: Counter = Counter()   # overall frequency
        opp_phase: dict = {}              # keyed by round_index % 6
        opp_after: dict = {}             # keyed by opponent's last move
        opp_bigram: dict = {}            # keyed by (prev_prev, prev) opponent moves
        op_last = op_prev = None
        n_rounds = len(history)

        if opp_name and n_rounds > 0:
            opp_seq = [str(m) for m in history[opp_name].values]
            op_last = opp_seq[-1]
            op_prev = opp_seq[-2] if n_rounds >= 2 else None
            for i, m in enumerate(opp_seq):
                opp_global[m] += 1
                opp_phase.setdefault(i % 6, Counter())[m] += 1
                if i > 0:
                    opp_after.setdefault(opp_seq[i - 1], Counter())[m] += 1
                if i > 1:
                    opp_bigram.setdefault((opp_seq[i - 2], opp_seq[i - 1]), Counter())[m] += 1

        def probs(counts: Counter, alpha: float = 1.0) -> dict:
            total = sum(counts.get(m, 0) for m in MOVES) + alpha * 5
            return {m: (counts.get(m, 0) + alpha) / total for m in MOVES}

        p_global = probs(opp_global, alpha=2.0)
        p_phase = probs(opp_phase.get(n_rounds % 6, Counter()), alpha=1.0)
        p_after = probs(opp_after.get(op_last, Counter()), alpha=1.0) if op_last else {m: 0.2 for m in MOVES}
        bigram_key = (op_prev, op_last) if op_prev and op_last else None
        bigram_data = opp_bigram.get(bigram_key, Counter()) if bigram_key else Counter()
        bigram_n = sum(bigram_data.values())
        p_bigram = probs(bigram_data, alpha=1.0) if bigram_n >= 5 else {m: 0.2 for m in MOVES}

        # Blend signals: weights increase as data accumulates
        w_global = 0.35
        w_phase = 0.15
        w_after = float(min(0.35, 0.08 + n_rounds / 800))
        w_bigram = float(min(0.15, bigram_n / 80)) if bigram_n >= 5 else 0.0
        w_sum = w_global + w_phase + w_after + w_bigram or 1.0
        op_dist = {
            m: (w_global * p_global[m] + w_phase * p_phase[m]
                + w_after * p_after[m] + w_bigram * p_bigram[m]) / w_sum
            for m in MOVES
        }

        # Enforce gun feasibility constraint
        if not opp_can_fire:
            op_dist["g"] = 0.0
            non_g = sum(op_dist[m] for m in "rpsd") or 1.0
            for m in "rpsd":
                op_dist[m] /= non_g
        else:
            # Opponent can fire: boost gun probability
            hist_rate = opp_global.get("g", 0) / max(sum(opp_global.values()), 1)
            boosted_g = max(0.6, hist_rate)
            non_g_total = sum(op_dist[m] for m in "rpsd") or 1.0
            scale = (1.0 - boosted_g) / non_g_total
            for m in "rpsd":
                op_dist[m] *= scale
            op_dist["g"] = boosted_g

        # --- Softmax action selection (reduces exploitability) ---
        legal = ["r", "p", "s", "d"]
        if can_fire:
            legal.append("g")

        utils = {m: sum(WIN[(m, opp)] * op_dist[opp] for opp in MOVES) for m in legal}
        tau = max(0.07, 0.3 - n_rounds / 1500)  # temperature cools over game
        max_u = max(utils.values())
        weights_sm = {m: float(np.exp((utils[m] - max_u) / tau)) for m in legal}
        z = sum(weights_sm.values())
        policy = [weights_sm[m] / z for m in legal]
        chosen = str(np.random.choice(legal, p=policy))

        # --- Urgency-scaled Kelly betting ---
        p_win = p_lose = 0.0
        for opp, p in op_dist.items():
            wv = WIN.get((chosen, opp), 0)
            if wv > 0:
                p_win += p
            elif wv < 0:
                p_lose += p
            else:
                p_win += 0.5 * p
                p_lose += 0.5 * p

        edge = p_win - p_lose
        entropy = -sum(p * np.log(p + 1e-12) for p in op_dist.values()) / np.log(5)
        confidence = float(np.clip(1.0 - entropy, 0.0, 1.0))
        max_legal_bet = max(0, int(my_score)) + 2000

        if edge <= 0 or my_score < 0:
            bet = 0
        else:
            urgency = 1.0
            if rounds_remaining < 60:
                urgency = 1.4
            if rounds_remaining < 20:
                urgency = 2.0
            raw_bet = max_legal_bet * edge * (0.15 + 0.85 * confidence) * 0.6 * urgency
            bet = int(np.clip(raw_bet, 0, max_legal_bet))

        return (chosen, bet)

        

    def _minority_update_from_new_round(self, history: pd.DataFrame):
        current_idx = len(history) - 1
        if current_idx <= self.minority_last_round_idx:
            return 

        players = [col for col in history.columns if col != "rounds_remaining"]

        if not self.minority_players:
            self.minority_players = players
            for p in players:
                self.minority_prev_moves[p] = None
                self.minority_losses[p] = 0
                self.minority_switches[p] = 0
                self.minority_last_won[p] = True

        for i in range(self.minority_last_round_idx + 1, current_idx + 1):
            row = history.iloc[i]
            choices = row[self.minority_players]

            count_A = (choices == "A").sum()
            count_B = (choices == "B").sum()
            if count_A == count_B:
                minority_choice = None
            else:
                minority_choice = "A" if count_A < count_B else "B"

            for player in self.minority_players:
                move = choices[player]
                won = (move == minority_choice) if minority_choice else False

                if not self.minority_last_won[player]:
                    if self.minority_prev_moves[player] is not None and self.minority_prev_moves[player] != move:
                        self.minority_switches[player] += 1
                    self.minority_losses[player] += 1

                self.minority_prev_moves[player] = move
                self.minority_last_won[player] = won

        self.minority_last_round_idx = current_idx

    def minority(self, history: pd.DataFrame) -> Literal["A", "B"]:
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
        if history.empty or len(history) < 2:
            return np.random.choice(["A", "B"])

        self._minority_update_from_new_round(history)

        predictions = []
        for player in self.minority_players:
            last_move = self.minority_prev_moves[player]
            if last_move is None:
                predictions.append(np.random.choice(["A", "B"]))
                continue

            if not self.minority_last_won[player]:
                loss_count = self.minority_losses[player]
                switch_count = self.minority_switches[player]
                switch_rate = switch_count / loss_count if loss_count > 0 else 0.5
                if switch_rate > 0.6:
                    predicted = "B" if last_move == "A" else "A"
                else:
                    predicted = last_move
            else:
                predicted = last_move

            predictions.append(predicted)

        count_A = predictions.count("A")
        count_B = predictions.count("B")

        return "A" if count_A < count_B else "B"
