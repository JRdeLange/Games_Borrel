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
        my_pos = None
        current_direction = None

        
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                if grid[i, j] in ['>', '<', '^', 'v']:
                    my_pos = (i, j)
                    if grid[i, j] == '>':
                        current_direction = 'right'
                    elif grid[i, j] == '<':
                        current_direction = 'left'
                    elif grid[i, j] == '^':
                        current_direction = 'up'
                    elif grid[i, j] == 'v':
                        current_direction = 'down'
                    break
            if my_pos:
                break
        
        if not my_pos:
            return "right"  # Fallback
        
        # Define direction vectors
        directions = {
            'up': (-1, 0),
            'down': (1, 0),
            'left': (0, -1),
            'right': (0, 1)
        }
        
        # Opposite directions (can't turn 180 degrees)
        opposites = {
            'up': 'down',
            'down': 'up',
            'left': 'right',
            'right': 'left'
        }
        
        def count_reachable_spaces(start_row, start_col):
            """Count empty spaces reachable from a position using BFS."""
            if start_row < 0 or start_row >= grid.shape[0]:
                return 0
            if start_col < 0 or start_col >= grid.shape[1]:
                return 0
            if grid[start_row, start_col] != ' ':
                return 0
            
            visited = set()
            queue = [(start_row, start_col)]
            count = 0
            
            while queue:
                row, col = queue.pop(0)
                
                if (row, col) in visited:
                    continue
                if row < 0 or row >= grid.shape[0]:
                    continue
                if col < 0 or col >= grid.shape[1]:
                    continue
                if grid[row, col] != ' ':
                    continue
                
                visited.add((row, col))
                count += 1
                
                # Add neighbors
                queue.append((row + 1, col))
                queue.append((row - 1, col))
                queue.append((row, col + 1))
                queue.append((row, col - 1))
            
            return count
        
        # Evaluate each direction
        best_move = None
        best_score = -1
        
        for move_name, (dr, dc) in directions.items():
            # Skip opposite direction (would just continue in current direction)
            if current_direction and move_name == opposites[current_direction]:
                continue
            
            # Calculate next position
            next_row = my_pos[0] + dr
            next_col = my_pos[1] + dc
            
            # Check if move is valid
            if next_row < 0 or next_row >= grid.shape[0]:
                continue
            if next_col < 0 or next_col >= grid.shape[1]:
                continue
            if grid[next_row, next_col] != ' ':
                continue
            
            # Count reachable spaces from this move
            score = count_reachable_spaces(next_row, next_col)
            
            if score > best_score:
                best_score = score
                best_move = move_name
        
        # If no valid move found (shouldn't happen), pick a random valid one
        if best_move is None:
            valid_moves = []
            for move_name, (dr, dc) in directions.items():
                next_row = my_pos[0] + dr
                next_col = my_pos[1] + dc
                if (0 <= next_row < grid.shape[0] and 
                    0 <= next_col < grid.shape[1] and 
                    grid[next_row, next_col] == ' '):
                    valid_moves.append(move_name)
            
            if valid_moves:
                best_move = str(np.random.choice(valid_moves))
            else:
                best_move = "right"  # Last resort
        

        # for the first rounds we set up a set strategy. 

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
        # Find all legal moves and pick a random one
        size = horizontal_lines.shape[1]
        legal_moves = []
        for row in range(size + 1):
            for col in range(size):
                if not horizontal_lines[row, col]:
                    legal_moves.append({"orientation": "h", "row": row, "col": col})
        for row in range(size):
            for col in range(size + 1):
                if not vertical_lines[row, col]:
                    legal_moves.append({"orientation": "v", "row": row, "col": col})
        return legal_moves[np.random.randint(0, len(legal_moves))]

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

        # Priority 1: finish a piece (land exactly on home index)
        for pn, pos in on_board.items():
            if (pos + dice_roll) % board_size == home_idx:
                return pn

        # Priority 2: capture an opponent piece
        for pn in range(1, 5):
            if pn in finished:
                continue
            if pn in at_home:
                if dice_roll != 6:
                    continue
                target = start_pos
            elif pn in on_board:
                target = (on_board[pn] + dice_roll) % board_size
            else:
                continue
            target_space = board.loc[target, "space"]
            if isinstance(target_space, pd.Series):
                target_space = target_space.iloc[0]
            if isinstance(target_space, str) and not target_space.startswith(self.name + "_"):
                return pn

        # Priority 3: deploy from home when dice is 6
        if dice_roll == 6:
            for pn in at_home:
                if pn not in finished:
                    return pn

        # Priority 4: fallback — move the first piece that is actually movable
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
        # Determine current round number
        current_round = len(history)
        
        # Check if gun is available
        if len(history) == 0:
            gun_available = True
        else:
            my_reload_timer = history[f"{self.name}_reload_timer"].iloc[-1]
            # If reload_timer <= 1, gun is available for this round
            gun_available = my_reload_timer <= 1
        
        # Strategy: Use gun on rounds 7, 12, 17, 22, 27... (if available)
        gun_rounds = {7 + 5*i for i in range(120)}  # Generates {7, 12, 17, 22, 27, ..., 602}
        
        if current_round in gun_rounds and gun_available:
            choice = "g"
        elif len(history) > 0:
            # Find opponent's column name
            opponent_name = None
            for col in history.columns:
                if col not in [self.name, f"{self.name}_score", f"{self.name}_reload_timer", 
                            f"{self.name}_bet", "rounds_remaining"] and \
                history[col].dtype == 'object':
                    opponent_name = col
                    break
            
            if opponent_name:
                opponent_last_move = history[opponent_name].iloc[-1]
                
                # Second-order counter logic: beat what beats opponent's last move
                # (excluding gun from intermediate counters)
                second_order_responses = {
                    'r': ['p', 's', 'r'],  # rock -> {d,p} -> {p,s} ∪ {s,r}
                    'p': ['r'],             # paper -> {s} -> {r}
                    's': ['d', 'p'],       # scissors -> {r} -> {d,p}
                    'g': ['p', 's'],       # gun -> {d} -> {p,s}
                    'd': ['s', 'r']        # duck -> {p,s} -> {s} ∪ {r}
                }
                
                if opponent_last_move in second_order_responses:
                    possible_moves = second_order_responses[opponent_last_move]
                    choice = str(np.random.choice(possible_moves))
                else:
                    choice = "p"  # Default to duck
            else:
                choice = "p"  # Default to duck if can't find opponent
        else:
            # First round, default to duck
            choice = "p"

        # round 6 should be duck
        if current_round == 6:
            choice = "p"
        
        # Conservative betting strategy
        bet = 100
        
        return (choice, bet)
        # return (str(np.random.choice(["r", "p", "s", "g", "d"])), 100)
