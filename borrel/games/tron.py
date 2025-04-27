import copy

import numpy as np

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


class Position:
    def __init__(self, x: int, y: int):
        self.position = np.array([x, y])

    def move(self, direction):
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
        self.grid = [[" " for _ in range(width)] for _ in range(height)]

    def place_character(self, position, character):
        if position.x < 0 or position.x >= self.width:
            return

        if position.y < 0 or position.y >= self.height:
            return
        # print(f"placing {character} at {position.x}, {position.y}")
        self.grid[position.x][position.y] = character

    def is_occupied(self, position):
        if position.x < 0 or position.x >= self.width:
            return True

        if position.y < 0 or position.y >= self.height:
            return True

        if self.grid[position.x][position.y] != " ":
            return True

        return False

    def from_perspective(self, perspective: "Player", opponent: "Player"):
        grid_copy = np.array(copy.deepcopy(self.grid))
        # grid_copy[perspective.position.x][perspective.position.y] = "O"
        grid_copy[opponent.position.x][opponent.position.y] = "X"

        persp_wall_mask = grid_copy == perspective.character
        grid_copy[persp_wall_mask] = "o"

        opp_wall_mask = grid_copy == opponent.character
        grid_copy[opp_wall_mask] = "x"

        # change all self.character to "o"
        return grid_copy


class Player:
    def __init__(
        self,
        grid: "Grid",
        position: Position,
        direction: np.ndarray,
        character: str,
        move: callable,
    ):
        self.grid = grid
        self.position = position
        self.direction = direction
        self.character = character
        self.alive = True
        self.move_function = move

    def set_opponent(self, opponent):
        self.opponent = opponent

    def take_turn(self):
        self.grid.move(self, self.direction)

    def is_dead(self):
        return not self.alive

    def turn(self, direction):
        # Can't turn 180 degrees
        if all(self.direction + direction == np.array([0, 0])):
            return

        self.direction = direction

    def choose_new_direction(self, opponent):
        self.turn(
            DIRDICT[self.move_function(self.grid.from_perspective(self, opponent))]
        )

    def move(self):
        self.position = self.position.move(self.direction)

    def leave_trail(self):
        self.grid.place_character(self.position, self.character)

    def check_collision(self):
        if self.grid.is_occupied(self.position):
            self.alive = False

    def draw_head(self):
        if not self.alive:
            self.grid.place_character(self.position, "X")
            return

        character = None
        if all(self.direction == RIGHT):
            character = ">"
        elif all(self.direction == LEFT):
            character = "<"
        elif all(self.direction == UP):
            character = "^"
        elif all(self.direction == DOWN):
            character = "v"

        assert character is not None

        self.grid.place_character(self.position, character)


class Game:
    def __init__(self, width: int, height: int, players: list):
        self.grid = Grid(height, width)

        self.players = [
            Player(
                self.grid,
                Position(height // 2, width // 2 - (width // 4)),
                RIGHT,
                "-",
                players[0].tron,
            ),
            Player(
                self.grid,
                Position(height // 2, width // 2 + (width // 4)),
                LEFT,
                "|",
                players[1].tron,
            ),
        ]

        for player in self.players:
            player.draw_head()

    def step(self):
        for i, player in enumerate(self.players):
            # choose new direction with equal info
            player.choose_new_direction(self.players[1 - i])

        for player in self.players:
            # leave trail where the head was and set new head position (do not draw yet)
            player.leave_trail()
            player.move()

        # Check for bike collision (heads at same pos)
        if self.players[0].position == self.players[1].position:
            self.players[0].alive = False
            self.players[1].alive = False

            self.players[0].draw_head()
            return False

        for player in self.players:
            # Check for collision with walls or trails (heads are not drawn yet)
            player.check_collision()

        for player in self.players:
            # Draw the head of the bike (if dead, draw X, else draw direction)
            player.draw_head()

        if self.players[0].is_dead() or self.players[1].is_dead():
            return False

        return True


class Renderer:
    def __init__(self, game: Game):
        self.game = game

    def render(self):
        print("+" * (len(self.game.grid.grid[0] * 2) + 3))
        for row in self.game.grid.grid:
            print("+ " + " ".join(row) + " +")

        print("+" * (len(self.game.grid.grid[0] * 2) + 3))


class Tron:
    def __init__(self, players: list, size: int = 7, render: bool = False):
        self.size = size
        self.render_state = render
        assert len(players) == 2, "There must be exactly two players."
        assert (
            len(set([player.name for player in players])) == 2
        ), "Player names must be unique."
        self.players = players
        self.game = Game(self.size, self.size, self.players)
        self.renderer = Renderer(self.game)

        if self.render_state:
            self.renderer.render()

    def step(self):
        keep_going = self.game.step()
        if self.render_state:
            self.renderer.render()
        return keep_going

    def play_game(self):
        keep_going = True
        while keep_going:
            keep_going = self.step()

        # Find out who won, if any
        if self.game.players[0].is_dead() and self.game.players[1].is_dead():
            return "tie"
        elif self.game.players[0].is_dead():
            return self.players[1].name
        elif self.game.players[1].is_dead():
            return self.players[0].name
        else:
            raise Exception("Game ended without a winner.")
