import random
from collections import deque
from typing import Literal

import numpy as np
import pandas as pd
from collections import deque
import random

class Ivo:

    def __init__(self, name: str = "ivo"):
        # NOTE: DO NOT TOUCH!
        self.name = name  # NOTE: DO NOT TOUCH!
        # NOTE: DO NOT TOUCH!

        # Feel free to store whatever you want here.
        # Each full run of a game will use a fresh instance of this class.
        # So for for example battleship, a new instance will be created for each game, but not for each turn.

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

        if len(history) == 0:
            # if this is the first round, return a random choice
            return str(np.random.choice(["A", "B"]))

        strategy = "ε-greedy"  # or "random", or "win-shift-lose-stay"
        if strategy == "win-stay-lose-shift":
            my_choice = history[self.name].iloc[-1]
            if self.detect_minority_win(history):
                # if I won, I will stay with my choice
                return my_choice
            else:
                return "A" if my_choice == "B" else "B"
        elif strategy == "win-shift-lose-stay":
            my_choice = history[self.name].iloc[-1]
            if self.detect_minority_win(history):
                # if I won, I will shift my choice
                return "A" if my_choice == "B" else "B"
            else:
                return my_choice
        elif strategy == "random":
            # if I won, I will stay with my choice
            return str(np.random.choice(["A", "B"]))

        elif strategy == "ε-greedy":
            ε = 0.1
            # 1) pure exploration
            if len(history) == 0 or np.random.rand() < ε:
                return np.random.choice(["A", "B"])
            # 2) exploitation: Win-Stay-Lose-Shift
            last = history[self.name].iloc[-1]
            # did you actually “win” on that last choice?
            # minority = the choice with fewer players in the last row
            counts = history.iloc[-1].value_counts()
            minority_choice = "A" if counts.get("A",0) < counts.get("B",0) else "B"
            won_last = (last == minority_choice)
            if won_last:
                return last
            else:
                return "A" if last == "B" else "B"


        print("ERROR: Unknown strategy. Please tell Ivo he has a bug.")
        return str(np.random.choice(["A", "B"]))

    def get_other_move(self, move):
        """
        Returns the move of the other player.
        """
        if move == "A":
            return "B"
        elif move == "B":
            return "A"
        else:
            raise ValueError("Invalid move")

    def detect_minority_win(self, history: pd.DataFrame) -> bool:
        # grab the last row of the history dataframe
        last_row = history.iloc[-1]
        # get the number of players who chose A and B
        num_A = (last_row == "A").sum()
        num_B = (last_row == "B").sum()
        if num_A == 0 or num_B == 0:
            return False
        # check if the player won
        if num_A < num_B:
            winner = "A"
        else:
            winner = "B"
        # check if the player is in the minority
        if last_row[self.name] == winner:
            return True
        else:
            return False

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

        # Thanks Carlos :)

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

        legal_moves = []
        for move, (dr, dc) in self.tron_dirs.items():
            nr, nc = self_pos[0] + dr, self_pos[1] + dc
            if self._tron_is_safe(nr, nc):
                legal_moves.append(move)

        if not legal_moves:
            return self_dir

        best_move = max(
            legal_moves,
            key=lambda move: self._tron_flood_fill_area(
                self_pos[0] + self.tron_dirs[move][0],
                self_pos[1] + self.tron_dirs[move][1]
            )
        )

        return best_move


        # for move in ["up", "down", "left", "right"]:
        #     if self.tron_check_valid_move(grid, move):
        #         # if the move is valid, return it
        #         return move
        # if no valid move is found, return a random move

        # return str(np.random.choice(["up", "down", "left", "right"]))

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


    def tron_check_valid_move(
        self, grid: np.ndarray, move: Literal["up", "down", "left", "right"]
    ) -> bool:
        """
        Check if the move is valid.
        A move is valid if it does not hit a wall or a bike, and does not leave the grid.
        """
        # get the current position of the bike
        bike_pos = np.argwhere(grid == "X")[0]
        # get the new position of the bike
        if move == "up":
            new_pos = (bike_pos[0] - 1, bike_pos[1])
        elif move == "down":
            new_pos = (bike_pos[0] + 1, bike_pos[1])
        elif move == "left":
            new_pos = (bike_pos[0], bike_pos[1] - 1)
        elif move == "right":
            new_pos = (bike_pos[0], bike_pos[1] + 1)
        else:
            raise ValueError("Invalid move")
        # check if the new position is valid
        if (
            new_pos[0] < 0
            or new_pos[0] >= grid.shape[0]
            or new_pos[1] < 0
            or new_pos[1] >= grid.shape[1]
            or grid[new_pos] in ["o", "x", "X"]
        ):
            return False
        return True

    def battleship_place_boats(
        self, boat_template: pd.DataFrame, grid_size: int
    ) -> pd.DataFrame:
        """
        This game is just Battleship (or in dutch: Zeeslag).
        First place your boats on the grid, then take turns firing at each other's grid.
        Once you destroy all boats of the opponent, you win!

        Your receive a pandas dataframe representing the boats to be placed.
        The dataframe has the following columns:
        - length: the length of the boat (int)
        - position: the position of the boat (list[int, int])
        - direction: the direction of the boat (Literal["up", "down", "left", "right"])

        Write a function that returns a pandas dataframe with the position and direction columns
        changed to whatever you want.

        So in a 3x3 grid:
        . . .
        . . .
        . . .

        A boat {length: 2, position: [1, 0], direction: "right"} would look like this:
        . . .
        S S .
        . . .

        Make sure the configuration is valid! Otherwise it is an instant loss :)

        The input here is always identical between games:

        boat_template:
           length position direction
        0       2   [0, 0]     right
        1       2   [1, 0]     right
        2       3   [2, 0]     right
        3       5   [3, 0]     right

        grid size is always 6x6

        Good luck with this age-old classic!
        """
        placement = boat_template.copy(deep=True)
        # occupancy map
        occupied = [[False]*grid_size for _ in range(grid_size)]
        directions = ["up", "down", "left", "right"]

        for idx, row in placement.iterrows():
            length = int(row["length"])
            while True:
                dir_ = random.choice(directions)
                # pick a random start so that boat of length L in dir_ stays on the board
                if dir_ == "up":
                    x = random.randint(length - 1, grid_size - 1)
                    y = random.randint(0, grid_size - 1)
                    coords = [(x - i, y) for i in range(length)]
                elif dir_ == "down":
                    x = random.randint(0, grid_size - length)
                    y = random.randint(0, grid_size - 1)
                    coords = [(x + i, y) for i in range(length)]
                elif dir_ == "left":
                    x = random.randint(0, grid_size - 1)
                    y = random.randint(length - 1, grid_size - 1)
                    coords = [(x, y - i) for i in range(length)]
                else:  # right
                    x = random.randint(0, grid_size - 1)
                    y = random.randint(0, grid_size - length)
                    coords = [(x, y + i) for i in range(length)]

                # check for overlaps
                if all(not occupied[r][c] for r, c in coords):
                    # mark these cells occupied
                    for r, c in coords:
                        occupied[r][c] = True

                    # record the chosen start & direction
                    placement.at[idx, "position"] = [coords[0][0], coords[0][1]]
                    placement.at[idx, "direction"] = dir_
                    break

        return placement

    def is_battleship_valid_placement(
        self, length, position, direction, grid_size: int
    ) -> pd.DataFrame:
        # check if the boat fits in the grid
        if direction == "up":
            if position[0] - length < 0:
                return False
        elif direction == "down":
            if position[0] + length >= grid_size:
                return False
        elif direction == "left":
            if position[1] - length < 0:
                return False
        elif direction == "right":
            if position[1] + length >= grid_size:
                return False
        else:
            raise ValueError("Invalid direction")

    def battleship_turn(
        self, own_fleet: np.ndarray, opponent_fleet: np.ndarray, grid_size: int
    ) -> list[int, int]:
        """
        Receives 2 grids (2d numpy arrays), representing your own fleet and the fleet of the opponent.
        On each grid, the following characters are present:
        - " " represents an empty cell
        - "S" represents a ship (only visible on your own fleet, hidden on opponent fleet)
        - "X" represents a hit
        - "o" represents a miss

        Write a function that returns a list of 2 integers representing the position to shoot at.

        Example turn (with full visibility)
        do shot at 4, 1
        dummy shot at 2, 2
        do's fleet:
        + + + + + + + +
        + S S         +
        + S S         +
        + S S X       +
        + S S S S S   +
        +             +
        +             +
        + + + + + + + +
        dummy's fleet:
        + + + + + + + +
        + S S         +
        + S S         +
        + S S S       +
        + S S S S S   +
        +   o         +
        +             +
        + + + + + + + +

        The grid size is always the same between games (6x6)

        Good luck with this age-old classic!
        """

            # If there is a hit, shoot around that hit
        def get_clusters(hits):
            clusters, seen = [], set()
            for h in hits:
                if h in seen:
                    continue
                queue = [h]
                seen.add(h)
                cluster = []
                while queue:
                    x, y = queue.pop()
                    cluster.append((x, y))
                    for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
                        nx, ny = x+dx, y+dy
                        if (nx, ny) in hits and (nx, ny) not in seen:
                            seen.add((nx, ny))
                            queue.append((nx, ny))
                clusters.append(cluster)
            return clusters

        # --- helper to find all empty neighbors, extending hits along orientation ---
        def get_adjacent(cluster):
            candidates = set()
            if len(cluster) > 1:
                xs = [x for x, _ in cluster]
                ys = [y for _, y in cluster]
                if len(set(xs)) == 1:
                    # horizontal
                    r = xs[0]
                    for c in (min(ys)-1, max(ys)+1):
                        if 0 <= c < grid_size and opponent_fleet[r, c] == " ":
                            candidates.add((r, c))
                else:
                    # vertical
                    c = ys[0]
                    for r in (min(xs)-1, max(xs)+1):
                        if 0 <= r < grid_size and opponent_fleet[r, c] == " ":
                            candidates.add((r, c))
            else:
                # single hit: all four directions
                x, y = cluster[0]
                for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
                    nx, ny = x+dx, y+dy
                    if 0 <= nx < grid_size and 0 <= ny < grid_size and opponent_fleet[nx, ny] == " ":
                        candidates.add((nx, ny))
            return list(candidates)

        # --- 1) reconstruct remaining ship lengths ---
        remaining = [2, 2, 3, 5]
        hits = list(zip(*np.where(opponent_fleet == "X")))
        clusters = get_clusters(hits)

        # any cluster whose length matches a ship AND has no blank extension is assumed sunk
        for cl in clusters:
            L = len(cl)
            if L in remaining:
                # check if it can still extend
                if not get_adjacent(cl):
                    remaining.remove(L)

        # --- 2) target mode: finish off any live cluster ---
        for cl in clusters:
            # only clusters with at least one adjacent blank are still “live”
            adj = get_adjacent(cl)
            if adj:
                # deterministic choice: smallest (row,col)
                x, y = sorted(adj)[0]
                return [x, y]

        # --- 3) hunt mode: simple parity scan ---
        for i in range(grid_size):
            for j in range(grid_size):
                if (i + j) % 2 == 0 and opponent_fleet[i, j] == " ":
                    return [i, j]

        # --- 4) fallback: first empty cell ---
        for i in range(grid_size):
            for j in range(grid_size):
                if opponent_fleet[i, j] == " ":
                    return [i, j]

        # no moves left
        return [0, 0]


    def wonky_rps(
        self,
        wonk_level: Literal[
            "wonk",
            "turbowonk",
            "hyperwonk",
            "golden wonk",
            "golden turbowonk",
            "golden hyperwonk",
        ],
        wonky_hand: Literal["r", "p", "s"],
        history: pd.DataFrame,
    ) -> Literal["r", "p", "s"]:
        """
        Rock, paper, scissors, but with a twist!

        Each round is just like normal rock paper scissors, with the usual rules:
        - rock beats scissors
        - scissors beats paper
        - paper beats rock
        - if both players choose the same hand, it is a tie

        If you win a round, you get 100 points. Most points at the end wins!

        However, each round one of the hands is chosen to be the "wonky hand".
        If you win with the wonky hand, your points are multiplied by an unknown "wonkfactor".
        A hand can be:
        - wonk level "wonk": Points are multiplied by a random wonkfactor between -1 and 2
        - wonk level "turbowonk" (~25% chance): Points are multiplied by a random wonkfactor between -10 and 20
        - wonk level "hyperwonk" (~5% chance): Points are multiplied by a random wonkfactor between -50 and 100

        Finally there is a 20% chance the wonk level is "golden".
        In this case, the wonkfactor is the absolute of what it would have been - so always positive!.

        Each round you also get the complete history of the current game in a pandas dataframe.
        It has a row for each played round.
        It has the following columns:
        - `your name`: Your choice in this round ("r", "p", or "s")
        - `opponents name`: the choice of your opponent in this round ("r", "p", or "s")
        - `your name`_score: Your score after this round (int)
        - `opponents name`_score: the score of your opponent after this round (int)
        - wonk_level: the wonk level of this round ("wonk", "turbowonk", "hyperwonk", "golden wonk", "golden turbowonk", "golden hyperwonk")
        - wonky_hand: the wonky hand of this round ("r", "p", or "s")
        - wonkfactor: the wonk factor of this round (float)
        - rounds_remaining: the number of rounds remaining in the game (int)

        Note that `your name` and `opponents name` are the names of the players in the game and they and their order can change from game to game.

        Example:

          do dummy do_score dummy_score   wonk_level wonky_hand  wonkfactor rounds_remaining
        0  p     r      100           0    hyperwonk          s  -21.798346                4
        1  p     p      100           0  golden wonk          r    0.527926                3
        2  s     s      100           0         wonk          s    0.839399                2
        3  r     p      100        -700    turbowonk          p   -7.004510                1
        4  r     p      100        -600    hyperwonk          s   -6.366723                0

        Write a function that returns "r", "p", or "s" based on the current state of the game.

        A full game is always 400 turns.

        Good luck with this fun game of risk management!
        """

        golden_round = wonk_level.startswith("golden")
        if golden_round:
            return wonky_hand
        elif wonk_level == "wonk":
            return self.rps_get_winner(wonky_hand)
        else:
            return str(np.random.choice(["r", "p", "s"]))

        # return str(np.random.choice(["r", "p", "s"]))

    def rps_get_winner(self, hand: Literal["r", "p", "s"]) -> Literal["r", "p", "s"]:
        """
        Returns the winning hand of the given hand.
        """
        if hand == "r":
            return "p"
        elif hand == "p":
            return "s"
        elif hand == "s":
            return "r"
        else:
            raise ValueError("Invalid hand")
