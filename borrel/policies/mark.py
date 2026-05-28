from typing import Literal

import numpy as np
import pandas as pd


class Mark:
    def __init__(self, name: str = "mark"):
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
        rows, cols = grid.shape

        def in_bounds(pos: tuple[int, int]) -> bool:
            rr, cc = pos
            return 0 <= rr < rows and 0 <= cc < cols

        direction_vectors = {
            "up": np.array([-1, 0]),
            "down": np.array([1, 0]),
            "left": np.array([0, -1]),
            "right": np.array([0, 1]),
        }
        my_bike = ["^", "v", "<", ">"]
        my_pos = np.asarray(np.where(np.isin(grid, my_bike))).flatten()

        legal_moves = []
        for move, vector in direction_vectors.items():
            nxt = my_pos + vector
            if in_bounds(nxt) and grid[tuple(nxt)] == " ":
                legal_moves.append(move)

        try:
            return str(np.random.choice(legal_moves))
        except:
            return "up"

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
        # Strategy notes:
        # Both Mees and Dominique greedy-take every chain offered to them and never
        # use the double-cross (hard-hearted handout). The classic counter is:
        #   1. Take free boxes (closing moves that don't open a new chain).
        #   2. When we're walking a chain and only two boxes are left, play the
        #      "domino" move that hands those two boxes to the opponent instead of
        #      closing them, forcing the opponent to open the next chain.
        #   3. When forced to sacrifice, give the opponent the smallest chain so
        #      we keep chain-parity control.
        # We also account for the fact that those opponents greedily take whole
        # chains, so any sacrifice we hand them will be entirely consumed.
        size = horizontal_lines.shape[1]
        H = np.asarray(horizontal_lines).copy()
        V = np.asarray(vertical_lines).copy()

        def sides_count(h: np.ndarray, v: np.ndarray, br: int, bc: int) -> int:
            return (
                int(h[br, bc])
                + int(h[br + 1, bc])
                + int(v[br, bc])
                + int(v[br, bc + 1])
            )

        def affected_boxes(m: tuple) -> list[tuple[int, int]]:
            o, r, c = m
            out: list[tuple[int, int]] = []
            if o == "h":
                if r > 0:
                    out.append((r - 1, c))
                if r < size:
                    out.append((r, c))
            else:
                if c > 0:
                    out.append((r, c - 1))
                if c < size:
                    out.append((r, c))
            return out

        def list_legal(h: np.ndarray, v: np.ndarray) -> list[tuple]:
            out = []
            for r in range(size + 1):
                for c in range(size):
                    if not h[r, c]:
                        out.append(("h", r, c))
            for r in range(size):
                for c in range(size + 1):
                    if not v[r, c]:
                        out.append(("v", r, c))
            return out

        def play_move(
            h: np.ndarray, v: np.ndarray, m: tuple
        ) -> tuple[np.ndarray, np.ndarray]:
            h2 = h.copy()
            v2 = v.copy()
            if m[0] == "h":
                h2[m[1], m[2]] = True
            else:
                v2[m[1], m[2]] = True
            return h2, v2

        def n_closes(h: np.ndarray, v: np.ndarray, m: tuple) -> int:
            h2, v2 = play_move(h, v, m)
            return sum(1 for b in affected_boxes(m) if sides_count(h2, v2, *b) == 4)

        def n_creates_3rd(h: np.ndarray, v: np.ndarray, m: tuple) -> int:
            h2, v2 = play_move(h, v, m)
            return sum(1 for b in affected_boxes(m) if sides_count(h2, v2, *b) == 3)

        def find_3sided(h: np.ndarray, v: np.ndarray) -> list[tuple[int, int]]:
            out = []
            for br in range(size):
                for bc in range(size):
                    if sides_count(h, v, br, bc) == 3:
                        out.append((br, bc))
            return out

        def closing_move_for(
            h: np.ndarray, v: np.ndarray, br: int, bc: int
        ) -> tuple:
            if not h[br, bc]:
                return ("h", br, bc)
            if not h[br + 1, bc]:
                return ("h", br + 1, bc)
            if not v[br, bc]:
                return ("v", br, bc)
            return ("v", br, bc + 1)

        def greedy_chain_take(
            h: np.ndarray, v: np.ndarray
        ) -> tuple[int, np.ndarray, np.ndarray]:
            """Mimic the greedy opponent: close every 3-sided box until none remain."""
            h = h.copy()
            v = v.copy()
            taken = 0
            while True:
                t = find_3sided(h, v)
                if not t:
                    return taken, h, v
                m = closing_move_for(h, v, *t[0])
                if m[0] == "h":
                    h[m[1], m[2]] = True
                else:
                    v[m[1], m[2]] = True
                for b in affected_boxes(m):
                    if sides_count(h, v, *b) == 4:
                        taken += 1

        def to_dict(m: tuple) -> dict:
            return {"orientation": m[0], "row": int(m[1]), "col": int(m[2])}

        all_moves = list_legal(H, V)
        if not all_moves:
            return {"orientation": "h", "row": 0, "col": 0}

        threes = find_3sided(H, V)

        # =====================================================================
        # Case A: at least one 3-sided box exists. Decide between greedy take
        # and a double-cross "domino" move.
        # =====================================================================
        if threes:
            greedy_gain, h_after_greedy, v_after_greedy = greedy_chain_take(H, V)
            moves_after_greedy = list_legal(h_after_greedy, v_after_greedy)

            # If the chain we'd take ends the game, just take it.
            if not moves_after_greedy:
                return to_dict(closing_move_for(H, V, *threes[0]))

            # What does greedy-then-required-move cost us?
            safe_after_greedy = [
                m
                for m in moves_after_greedy
                if n_creates_3rd(h_after_greedy, v_after_greedy, m) == 0
            ]
            if safe_after_greedy:
                # We can stop with a safe move; greedy is essentially free.
                my_sacrifice_cost = 0
            else:
                costs = []
                for m in moves_after_greedy:
                    h2, v2 = play_move(h_after_greedy, v_after_greedy, m)
                    opp_gain, _, _ = greedy_chain_take(h2, v2)
                    costs.append(opp_gain)
                my_sacrifice_cost = min(costs) if costs else 0

            greedy_net = greedy_gain - my_sacrifice_cost

            # Look for the strongest double-cross / domino move available now.
            best_dc_net = greedy_net
            best_dc_move: tuple | None = None

            for m in all_moves:
                if n_closes(H, V, m) > 0:
                    continue  # closing move, not a DC candidate
                if n_creates_3rd(H, V, m) == 0:
                    continue  # cannot be a domino — must create a new 3-sided box
                h_m, v_m = play_move(H, V, m)
                opp_take, h_oa, v_oa = greedy_chain_take(h_m, v_m)

                # A clean double-cross hands the opponent exactly 2 boxes (or 4
                # for a loop double-cross). Anything bigger is just feeding
                # them a chain.
                if opp_take not in (2, 4):
                    continue

                moves_after_opp = list_legal(h_oa, v_oa)
                if not moves_after_opp:
                    continue

                opp_has_safe = any(
                    n_creates_3rd(h_oa, v_oa, mm) == 0 for mm in moves_after_opp
                )
                if opp_has_safe:
                    # Opponent escapes with a safe move — DC just gave away boxes.
                    continue

                # Opponent is forced to sacrifice. They pick the cheapest one
                # for them, which is the smallest chain for us.
                opp_sacs = []
                for mm in moves_after_opp:
                    h2, v2 = play_move(h_oa, v_oa, mm)
                    our_gain_back, _, _ = greedy_chain_take(h2, v2)
                    opp_sacs.append(our_gain_back)
                our_take_back = min(opp_sacs) if opp_sacs else 0

                dc_net = -opp_take + our_take_back
                if dc_net > best_dc_net:
                    best_dc_net = dc_net
                    best_dc_move = m

            if best_dc_move is not None:
                return to_dict(best_dc_move)

            # No profitable double-cross: take the chain greedily, preferring
            # moves that don't unnecessarily extend the opponent's future take.
            scoring = [m for m in all_moves if n_closes(H, V, m) > 0]
            best_take: tuple | None = None
            best_take_score = -1e18
            for m in scoring:
                h2, v2 = play_move(H, V, m)
                # After taking, simulate the rest of the chain greedily and
                # see how much the opponent would get when we hand off.
                extra, h3, v3 = greedy_chain_take(h2, v2)
                ms3 = list_legal(h3, v3)
                if ms3:
                    safe3 = any(n_creates_3rd(h3, v3, mm) == 0 for mm in ms3)
                    if safe3:
                        opp_followup = 0
                    else:
                        followups = []
                        for mm in ms3:
                            h4, v4 = play_move(h3, v3, mm)
                            og, _, _ = greedy_chain_take(h4, v4)
                            followups.append(og)
                        opp_followup = min(followups) if followups else 0
                else:
                    opp_followup = 0
                score = (
                    n_closes(H, V, m)
                    + extra
                    - opp_followup
                    + float(np.random.uniform(0, 1e-3))
                )
                if score > best_take_score:
                    best_take_score = score
                    best_take = m

            if best_take is not None:
                return to_dict(best_take)
            return to_dict(closing_move_for(H, V, *threes[0]))

        # =====================================================================
        # Case B: no 3-sided box. Play safe if possible, otherwise hand over
        # the smallest possible sacrifice.
        # =====================================================================
        safe = [m for m in all_moves if n_creates_3rd(H, V, m) == 0]

        if safe:
            best_move: tuple | None = None
            best_score = -1e18
            # We want to be the player NOT forced to open the first long chain.
            # Heuristic: pick the safe move that leaves the largest pool of
            # remaining safe moves, with a tiny center bias for tie-breaking.
            for m in safe:
                h2, v2 = play_move(H, V, m)
                ms2 = list_legal(h2, v2)
                future_safe = sum(
                    1 for mm in ms2 if n_creates_3rd(h2, v2, mm) == 0
                )
                # Tie-break: prefer edges (lower future "two-sided" structures
                # which can blow up into chains we control).
                two_sided = sum(
                    1
                    for br in range(size)
                    for bc in range(size)
                    if sides_count(h2, v2, br, bc) == 2
                )
                if m[0] == "h":
                    center_dist = abs(m[1] - size / 2.0) + abs(
                        m[2] - (size - 1) / 2.0
                    )
                else:
                    center_dist = abs(m[1] - (size - 1) / 2.0) + abs(
                        m[2] - size / 2.0
                    )
                score = (
                    10.0 * future_safe
                    - 0.05 * two_sided
                    - 0.1 * center_dist
                    + float(np.random.uniform(0, 1e-3))
                )
                if score > best_score:
                    best_score = score
                    best_move = m
            if best_move is not None:
                return to_dict(best_move)

        # All moves create a 3-sided box. Pick the smallest sacrifice.
        # Important: the modelled opponents (Mees, Dominique) greedily eat the
        # whole chain we hand them, so the "cost" of a sacrifice is the size of
        # the chain they will consume. If after their take they are forced to
        # sacrifice back to us, we credit ourselves the smallest chain they
        # could give.
        best_move = None
        best_net_loss = 1e18
        for m in all_moves:
            h2, v2 = play_move(H, V, m)
            opp_gain, h3, v3 = greedy_chain_take(h2, v2)
            ms_after = list_legal(h3, v3)
            if not ms_after:
                net_loss = float(opp_gain)
            else:
                opp_safe_after = any(
                    n_creates_3rd(h3, v3, mm) == 0 for mm in ms_after
                )
                if opp_safe_after:
                    net_loss = float(opp_gain)
                else:
                    our_takes = []
                    for mm in ms_after:
                        h4, v4 = play_move(h3, v3, mm)
                        ot, _, _ = greedy_chain_take(h4, v4)
                        our_takes.append(ot)
                    our_take_back = min(our_takes) if our_takes else 0
                    net_loss = float(opp_gain - our_take_back)
            net_loss += float(np.random.uniform(0, 1e-3))
            if net_loss < best_net_loss:
                best_net_loss = net_loss
                best_move = m

        if best_move is None:
            best_move = all_moves[0]
        return to_dict(best_move)

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
        board_size = len(board)
        my_home_position = (board["home"] == self.name).idxmax()

        if dice_roll == 6 and len(info["pieces_at_home"][self.name]) > 0:
            return info["pieces_at_home"][self.name][0]

        for nr in range(1, 5):
            if (
                nr not in info["pieces_finished"][self.name]
                and nr not in info["pieces_at_home"][self.name]
            ):
                current_position = (board["space"] == f"{self.name}_{nr}").idxmax()
                new_position = (current_position + dice_roll) % board_size

                if new_position == my_home_position:
                    return nr
                if board.loc[new_position, "space"] is not None:
                    return nr

        # Dummy policy for sorry, just keeps moving the first piece that is not finished yet
        for nr in range(1, 5):
            if nr not in info["pieces_finished"][self.name]:
                return nr

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
        choices = {"s": 1, "p": 1, "r": 0.5}

        if len(history) == 0:
            is_my_gun_legal = False
            is_opponents_gun_legal = False
        else:
            if history.columns[0] == self.name:
                opponent_name = history.columns[1]
            else:
                opponent_name = history.columns[0]
            is_my_gun_legal = history["mark_reload_timer"].iloc[-1] == 0
            is_opponents_gun_legal = (
                history[f"{opponent_name}_reload_timer"].iloc[-1] == 0
            )

        if is_my_gun_legal:
            choices["g"] = 2

        if is_opponents_gun_legal:
            choices["d"] = 0.5
            choices["r"] += 1

        amount = 100

        if is_my_gun_legal and not is_opponents_gun_legal:
            amount = 500

        probs = np.array(list(choices.values()))
        probs /= probs.sum()

        return str(
            np.random.choice(list(choices.keys()), p=probs
        )), amount

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
