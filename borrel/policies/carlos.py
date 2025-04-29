from typing import Literal

import numpy as np
import pandas as pd


class Carlos:

    def __init__(self, name: str = "carlos"):
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
        return str(np.random.choice(["A", "B"]))

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
        return str(np.random.choice(["up", "down", "left", "right"]))

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
        return boat_template

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
        x = np.random.randint(0, grid_size)
        y = np.random.randint(0, grid_size)
        return [x, y]

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
        return str(np.random.choice(["r", "p", "s"]))
