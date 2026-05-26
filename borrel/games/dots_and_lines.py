import copy

import numpy as np
import pandas as pd

HORIZONTAL = "h"
VERTICAL = "v"

ORIENTATION_MAP = {
	"h": HORIZONTAL,
	"horizontal": HORIZONTAL,
	"-": HORIZONTAL,
	"v": VERTICAL,
	"vertical": VERTICAL,
	"|": VERTICAL,
}


class Player:
	def __init__(self, policy_player):
		self.name = policy_player.name
		move_function = getattr(policy_player, "dots_and_lines", None)
		self.move_function = move_function if callable(move_function) else None


class Renderer:
	def __init__(self, game: "DotsAndLines"):
		self.game = game

	def render(self):
		board_lines = []
		for row in range(self.game.size + 1):
			dot_row = []
			for col in range(self.game.size):
				dot_row.append(".")
				dot_row.append("---" if self.game.horizontal_lines[row, col] else "   ")
			dot_row.append(".")
			board_lines.append("".join(dot_row))

			if row == self.game.size:
				continue

			cell_row = []
			for col in range(self.game.size):
				cell_row.append("|" if self.game.vertical_lines[row, col] else " ")
				owner = self.game.box_owners[row, col]
				marker = owner[0].upper() if owner is not None else " "
				cell_row.append(f" {marker} ")
			cell_row.append("|" if self.game.vertical_lines[row, self.game.size] else " ")
			board_lines.append("".join(cell_row))

		print("\n".join(board_lines))
		print(
			"Scores: "
			+ " | ".join(
				[f"{player.name}={self.game.scores[player.name]}" for player in self.game.players]
			)
		)


class DotsAndLines:
	"""
	Classic dots-and-boxes style game for two players.

	How the board is represented:
	- Dots are implicit: there are (size + 1) x (size + 1) dots.
	- Horizontal edges are tracked in `horizontal_lines` with shape (size + 1, size).
	- Vertical edges are tracked in `vertical_lines` with shape (size, size + 1).
	- Box owners are tracked in `box_owners` with shape (size, size), where each cell
	  is either None or the name of the player that completed that box.

	Turn flow:
	1) Current player returns a move.
	2) Move is normalized and validated.
	3) The chosen line is drawn.
	4) Any newly completed boxes are assigned to the current player and scored.
	5) If at least one box is completed, the same player keeps the turn.
	6) Otherwise turn passes to the opponent.
	7) When all lines are drawn, the higher score wins, otherwise it is a tie.

	Invalid move behavior:
	- If a callback crashes, returns an invalid format, or chooses an illegal line,
	  that player immediately loses and the opponent wins.

	Policy callback expected signature:
		dots_and_lines(horizontal_lines, vertical_lines, box_owners, history)

	State-isolation guarantee:
	- Every callback receives deep-copied snapshots of all mutable game state.
	- This prevents one player's policy from mutating shared state that could affect
	  the other player's decision or later turns.

	Move formats accepted:
	- {"orientation": "h"|"v", "row": int, "col": int}
	- ["h"|"v", row, col]
	- [row, col, "h"|"v"]
	"""

	def __init__(self, player1, player2, size: int = 3, render: bool = False):
		self.size = size
		self.render = render

		assert isinstance(self.size, int) and self.size > 0, "Size must be a positive integer."
		assert player1 is not player2, "Players must be different instances."
		assert player1.name != player2.name, "Player names must be unique."

		self.players = [Player(player1), Player(player2)]
		self.total_lines = 2 * self.size * (self.size + 1)

		self.score_columns = [
			f"{self.players[0].name}_score",
			f"{self.players[1].name}_score",
		]
		self.renderer = Renderer(self)

		self.reset()

	def _empty_history(self):
		return pd.DataFrame(
			columns=[
				"player",
				"orientation",
				"row",
				"col",
				"boxes_completed",
				self.score_columns[0],
				self.score_columns[1],
				"lines_remaining",
			]
		)

	def reset(self):
		self.horizontal_lines = np.zeros((self.size + 1, self.size), dtype=bool)
		self.vertical_lines = np.zeros((self.size, self.size + 1), dtype=bool)
		self.box_owners = np.full((self.size, self.size), None, dtype=object)
		self.scores = {player.name: 0 for player in self.players}
		self.history = self._empty_history()
		self.current_player_idx = int(np.random.randint(0, 2))

	def lines_drawn(self) -> int:
		return int(np.sum(self.horizontal_lines) + np.sum(self.vertical_lines))

	def lines_remaining(self) -> int:
		return self.total_lines - self.lines_drawn()

	def get_legal_moves(self) -> list[tuple[str, int, int]]:
		legal_moves = []

		for row in range(self.size + 1):
			for col in range(self.size):
				if not self.horizontal_lines[row, col]:
					legal_moves.append((HORIZONTAL, row, col))

		for row in range(self.size):
			for col in range(self.size + 1):
				if not self.vertical_lines[row, col]:
					legal_moves.append((VERTICAL, row, col))

		return legal_moves

	def _normalize_orientation(self, orientation):
		if not isinstance(orientation, str):
			return None

		return ORIENTATION_MAP.get(orientation.strip().lower())

	def _coerce_int(self, value):
		if isinstance(value, (int, np.integer)):
			return int(value)

		if isinstance(value, float) and value.is_integer():
			return int(value)

		return None

	def normalize_move(self, move):
		if isinstance(move, dict):
			orientation = move.get(
				"orientation", move.get("direction", move.get("axis"))
			)
			row = move.get("row", move.get("r", move.get("x")))
			col = move.get("col", move.get("c", move.get("y")))
		elif isinstance(move, (list, tuple, np.ndarray)) and len(move) == 3:
			if isinstance(move[0], str):
				orientation, row, col = move
			elif isinstance(move[2], str):
				row, col, orientation = move
			else:
				return None
		else:
			return None

		orientation = self._normalize_orientation(orientation)
		row = self._coerce_int(row)
		col = self._coerce_int(col)

		if orientation is None or row is None or col is None:
			return None

		return orientation, row, col

	def is_move_legal(self, orientation: str, row: int, col: int) -> bool:
		if orientation == HORIZONTAL:
			if row < 0 or row > self.size:
				return False
			if col < 0 or col >= self.size:
				return False
			return not self.horizontal_lines[row, col]

		if orientation == VERTICAL:
			if row < 0 or row >= self.size:
				return False
			if col < 0 or col > self.size:
				return False
			return not self.vertical_lines[row, col]

		return False

	def _is_box_complete(self, row: int, col: int) -> bool:
		return (
			self.horizontal_lines[row, col]
			and self.horizontal_lines[row + 1, col]
			and self.vertical_lines[row, col]
			and self.vertical_lines[row, col + 1]
		)

	def _candidate_boxes_from_move(self, orientation: str, row: int, col: int):
		boxes = []

		if orientation == HORIZONTAL:
			if row > 0:
				boxes.append((row - 1, col))
			if row < self.size:
				boxes.append((row, col))
		else:
			if col > 0:
				boxes.append((row, col - 1))
			if col < self.size:
				boxes.append((row, col))

		return boxes

	def apply_move(self, player_name: str, orientation: str, row: int, col: int) -> int:
		if orientation == HORIZONTAL:
			self.horizontal_lines[row, col] = True
		else:
			self.vertical_lines[row, col] = True

		completed_boxes = 0
		for box_row, box_col in self._candidate_boxes_from_move(orientation, row, col):
			if self.box_owners[box_row, box_col] is None and self._is_box_complete(
				box_row, box_col
			):
				self.box_owners[box_row, box_col] = player_name
				completed_boxes += 1

		if completed_boxes > 0:
			self.scores[player_name] += completed_boxes

		return completed_boxes

	def _record_turn(
		self,
		player_name: str,
		orientation: str,
		row: int,
		col: int,
		boxes_completed: int,
	):
		row_data = {
			"player": player_name,
			"orientation": orientation,
			"row": row,
			"col": col,
			"boxes_completed": boxes_completed,
			self.score_columns[0]: self.scores[self.players[0].name],
			self.score_columns[1]: self.scores[self.players[1].name],
			"lines_remaining": self.lines_remaining(),
		}
		self.history = pd.concat([self.history, pd.DataFrame([row_data])], ignore_index=True)

	def _fallback_random_move(self):
		legal_moves = self.get_legal_moves()
		if not legal_moves:
			return None

		return legal_moves[np.random.randint(0, len(legal_moves))]

	def _state_snapshot_for_callback(self):
		"""
		Return a fully isolated snapshot for policy callbacks.

		All arrays/dataframes are deep-copied so policy code cannot mutate live game
		state or data that another player will read later.
		"""
		return (
			copy.deepcopy(self.horizontal_lines),
			copy.deepcopy(self.vertical_lines),
			copy.deepcopy(self.box_owners),
			copy.deepcopy(self.history),
		)

	def _get_move(self, player: Player):
		if player.move_function is None:
			return self._fallback_random_move()

		try:
			move = player.move_function(*self._state_snapshot_for_callback())
		except Exception:
			return None

		return self.normalize_move(move)

	def get_winner(self):
		player_a = self.players[0].name
		player_b = self.players[1].name

		if self.scores[player_a] > self.scores[player_b]:
			return player_a
		if self.scores[player_b] > self.scores[player_a]:
			return player_b
		return "tie"

	def play_turn(self):
		current = self.players[self.current_player_idx]
		opponent = self.players[1 - self.current_player_idx]

		move = self._get_move(current)
		if move is None:
			return opponent.name

		orientation, row, col = move
		if not self.is_move_legal(orientation, row, col):
			return opponent.name

		boxes_completed = self.apply_move(current.name, orientation, row, col)
		self._record_turn(current.name, orientation, row, col, boxes_completed)

		if self.render:
			print(f"{current.name} plays {(orientation, row, col)}, completed {boxes_completed} box(es)")
			self.renderer.render()

		if self.lines_remaining() == 0:
			return self.get_winner()

		if boxes_completed == 0:
			self.current_player_idx = 1 - self.current_player_idx

		return "continue"

	def play_game(self):
		self.reset()

		if self.render:
			print(f"Starting player: {self.players[self.current_player_idx].name}")
			self.renderer.render()

		while True:
			result = self.play_turn()
			if result != "continue":
				return result
