import copy

import numpy as np
import pandas as pd

RIGHT = np.array([0, 1])
LEFT = np.array([0, -1])
UP = np.array([-1, 0])
DOWN = np.array([1, 0])

DIRDICT = {
    "up": UP,
    "down": DOWN,
    "left": LEFT,
    "right": RIGHT,
}

SHIPCHAR = "S"
HITCHAR = "X"
MISSCHAR = "o"
NOCHAR = " "


class Position:
    def __init__(self, x: int, y: int):
        self.position = np.array([x, y])

    def move(self, direction: np.array):
        return Position(self.x + direction[0], self.y + direction[1])

    @property
    def x(self):
        return self.position[0]

    @property
    def y(self):
        return self.position[1]

    def __eq__(self, other):
        if not isinstance(other, Position):
            return False
        return np.array_equal(self.position, other.position)


class Grid:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.grid = np.array([[NOCHAR for _ in range(width)] for _ in range(height)])

    def place_character(self, position: Position, character: str):
        if position.x < 0 or position.x >= self.width:
            return

        if position.y < 0 or position.y >= self.height:
            return

        self.grid[position.x][position.y] = character

    def place_boat(self, position: Position, direction: np.array, length: int):
        assert position.x >= 0 and position.x < self.width
        assert position.y >= 0 and position.y < self.height

        for i in range(length):
            if (position.x < 0 or position.x >= self.width) or (
                position.y < 0 or position.y >= self.height
            ):
                return False
            self.place_character(position, SHIPCHAR)
            position = position.move(direction)

        return True

    def place_boats(self, boats: pd.DataFrame):
        for i, boat in boats.iterrows():
            position = Position(boat["position"][0], boat["position"][1])
            direction = DIRDICT[boat["direction"]]
            length = boat["length"]
            if not self.place_boat(position, direction, length):
                return False
        return True

    def shot(self, position: Position):
        if position.x < 0 or position.x >= self.width:
            return

        if position.y < 0 or position.y >= self.height:
            return

        if (
            self.grid[position.x][position.y] == SHIPCHAR
            or self.grid[position.x][position.y] == HITCHAR
        ):
            self.grid[position.x][position.y] = HITCHAR
            return
        else:
            self.grid[position.x][position.y] = MISSCHAR
            return

    def all_boats_sunk(self):
        return not np.any(self.grid == SHIPCHAR)

    def print_grid(self):
        print("+ " * (self.width + 2))
        for row in self.grid:
            print("+ ", end="")
            for cell in row:
                print(cell, end="")
                print(" ", end="")
            print("+")
        print("+ " * (self.width + 2))

    def get_masked_grid(self):
        masked_grid = np.where(self.grid == SHIPCHAR, NOCHAR, self.grid)
        return masked_grid


class Player:
    def __init__(
        self,
        grid: Grid,
        place_boats: callable,
        turn: callable,
        name: str,
    ):
        self.grid = grid
        self.place_boats = place_boats
        self.turn = turn
        self.name = name


class Battleship:
    def __init__(
        self, player1: Player, player2: Player, size: int = 6, render: bool = False
    ):
        self.size = size
        self.render = render

        grid1 = Grid(self.size, self.size)
        grid2 = Grid(self.size, self.size)

        self.player1 = Player(
            grid1, player1.battleship_place_boats, player1.battleship_turn, player1.name
        )
        self.player2 = Player(
            grid2, player2.battleship_place_boats, player2.battleship_turn, player2.name
        )

    def place_boats(self):
        boat_template = pd.DataFrame(
            {
                "length": [2, 2, 3, 5],
                "position": [[0, 0], [1, 0], [2, 0], [3, 0]],
                "direction": ["right", "right", "right", "right"],
            }
        )

        player1_boats = self.player1.place_boats(
            copy.deepcopy(boat_template), self.size
        )
        player2_boats = self.player2.place_boats(
            copy.deepcopy(boat_template), self.size
        )

        if not player1_boats["length"].equals(boat_template["length"]):
            return self.player2.name

        if not player2_boats["length"].equals(boat_template["length"]):
            return self.player1.name

        placed_successfully1 = self.player1.grid.place_boats(player1_boats)
        placed_successfully2 = self.player2.grid.place_boats(player2_boats)

        if not placed_successfully1 and not placed_successfully2:
            return "tie"

        if not placed_successfully1:
            return self.player2.name

        if not placed_successfully2:
            return self.player1.name

        return "continue"

    def reset(self):
        grid1 = Grid(self.size, self.size)
        grid2 = Grid(self.size, self.size)
        self.player1.grid = grid1
        self.player2.grid = grid2

    def play_turn(self):
        # get the shots from both players
        shot1 = self.player1.turn(
            copy.deepcopy(self.player1.grid.grid),
            copy.deepcopy(self.player2.grid.get_masked_grid()),
            self.size,
        )
        shot2 = self.player2.turn(
            copy.deepcopy(self.player2.grid.grid),
            copy.deepcopy(self.player1.grid.get_masked_grid()),
            self.size,
        )

        shot1 = Position(shot1[0], shot1[1])
        shot2 = Position(shot2[0], shot2[1])

        # shoot
        self.player1.grid.shot(shot2)
        self.player2.grid.shot(shot1)

        if self.render:
            print(f"{self.player1.name} shot at {shot1.x}, {shot1.y}")
            print(f"{self.player2.name} shot at {shot2.x}, {shot2.y}")
            print(f"{self.player1.name}'s fleet:")
            self.player1.grid.print_grid()
            print(f"{self.player2.name}'s fleet:")
            self.player2.grid.print_grid()

        # check for wins
        if self.player1.grid.all_boats_sunk() and self.player2.grid.all_boats_sunk():
            return "tie"

        if self.player1.grid.all_boats_sunk():
            return self.player2.name

        if self.player2.grid.all_boats_sunk():
            return self.player1.name

        return "continue"

    def play_game(self):
        self.reset()
        winner = self.place_boats()
        if winner != "continue":
            return winner

        while True:
            winner = self.play_turn()
            if winner != "continue":
                return winner
