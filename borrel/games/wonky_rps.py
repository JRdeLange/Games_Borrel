import copy
import warnings

import numpy as np
import pandas as pd

ROCK = "r"
PAPER = "p"
SCISSORS = "s"

SCORE_BASE = 100


def get_wonkfactor():
    wonk_min = -1
    wonk_max = 2.0

    turbowonk_chance = 0.1
    turbowonk_factor = 10

    hyperwonk_chance = 0.01
    hyperwonk_factor = 50

    golden_wonk_chance = 0.1

    wonky_hand = np.random.choice([ROCK, PAPER, SCISSORS])

    base_wonkfactor = np.random.uniform(wonk_min, wonk_max)

    wonkfactor = base_wonkfactor
    wonk_level = "wonk"

    if np.random.rand() < turbowonk_chance:
        wonkfactor = base_wonkfactor * turbowonk_factor
        wonk_level = "turbowonk"

    if np.random.rand() < hyperwonk_chance:
        wonkfactor = base_wonkfactor * hyperwonk_factor
        wonk_level = "hyperwonk"

    if np.random.rand() < golden_wonk_chance:
        wonkfactor = abs(wonkfactor)
        wonk_level = "golden " + wonk_level

    return {
        "wonk_level": wonk_level,
        "wonky_hand": wonky_hand,
        "wonkfactor": wonkfactor,
    }


class WonkyRPS:
    def __init__(self, player1, player2, render: bool = False):
        self.player1 = player1
        self.player2 = player2
        self.render = render

        self.scores = {player1.name: 0, player2.name: 0}
        self.history = pd.DataFrame(
            columns=[
                player1.name,
                player2.name,
                player1.name + "_score",
                player2.name + "_score",
                "wonk_level",
                "wonky_hand",
                "wonkfactor",
                "rounds_remaining",
            ]
        )

    def determine_winner_loser(self, player1_choice, player2_choice):
        if player1_choice == player2_choice:
            return None

        if (
            (player1_choice == ROCK and player2_choice == SCISSORS)
            or (player1_choice == SCISSORS and player2_choice == PAPER)
            or (player1_choice == PAPER and player2_choice == ROCK)
        ):
            return self.player1
        else:
            return self.player2

    def get_history_from_perspective(self, perspective, opponent):
        """
        Returns a deep copy of the history dataframe, with the perspective of the given player.
        """
        perspective_history = copy.deepcopy(self.history)
        rename_dict = {
            perspective.name: "you",
            opponent.name: "opponent",
            perspective.name + "_score": "you_score",
            opponent.name + "_score": "opponent_score",
        }
        perspective_history.rename(columns=rename_dict, inplace=True)
        return perspective_history

    def play_round(self, rounds_remaining: int):
        wonk = get_wonkfactor()
        round_row = {
            "wonk_level": wonk["wonk_level"],
            "wonky_hand": wonk["wonky_hand"],
            "wonkfactor": wonk["wonkfactor"],
            "rounds_remaining": rounds_remaining,
        }

        player1_choice = self.player1.wonky_rps(
            wonk["wonk_level"],
            wonk["wonky_hand"],
            self.get_history_from_perspective(self.player1, self.player2),
        )
        if player1_choice not in [ROCK, PAPER, SCISSORS]:
            return self.player2.name

        player2_choice = self.player2.wonky_rps(
            wonk["wonk_level"],
            wonk["wonky_hand"],
            self.get_history_from_perspective(self.player2, self.player1),
        )
        if player2_choice not in [ROCK, PAPER, SCISSORS]:
            return self.player1.name

        round_row[self.player1.name] = player1_choice
        round_row[self.player2.name] = player2_choice

        # sorry for the ugly tree here
        winner = self.determine_winner_loser(player1_choice, player2_choice)
        if winner == self.player1:
            score = SCORE_BASE
            if player1_choice == wonk["wonky_hand"]:
                score *= wonk["wonkfactor"]

            self.scores[self.player1.name] += int(score)

        if winner == self.player2:
            score = SCORE_BASE
            if player2_choice == wonk["wonky_hand"]:
                score *= wonk["wonkfactor"]

            self.scores[self.player2.name] += int(score)

        round_row[self.player1.name + "_score"] = self.scores[self.player1.name]
        round_row[self.player2.name + "_score"] = self.scores[self.player2.name]

        # inside your play_round method, when updating self.history
        with warnings.catch_warnings():
            warnings.simplefilter(action="ignore", category=FutureWarning)
            self.history = pd.concat(
                [self.history, pd.DataFrame([round_row])], ignore_index=True
            )

        if self.render:
            print(
                f"Round {len(self.history)}: {self.player1.name} ({player1_choice}) vs {self.player2.name} ({player2_choice})"
            )
            print(
                f"{self.player1.name} score: {self.scores[self.player1.name]}, {self.player2.name} score: {self.scores[self.player2.name]}"
            )
            print(f"Wonk level: {wonk['wonk_level']}")
            print(f"Wonk factor: {wonk['wonkfactor']}")
            print(f"Wonky hand: {wonk['wonky_hand']}")
            print(
                f"Round result: {winner.name if winner else 'tie'}"
                + (f" gets {score} points" if winner else "")
            )
            print()

        return "continue"

    def reset(self):
        self.scores = {self.player1.name: 0, self.player2.name: 0}
        self.history = pd.DataFrame(
            columns=[
                self.player1.name,
                self.player2.name,
                self.player1.name + "_score",
                self.player2.name + "_score",
                "wonk_level",
                "wonky_hand",
                "wonkfactor",
                "rounds_remaining",
            ]
        )

    def play_game(self, rounds: int = 200):
        self.reset()

        for i in range(rounds):
            early_stop = self.play_round(rounds - i - 1)
            if early_stop != "continue":
                if self.render:
                    print("Invalid move, game over")
                return early_stop

        if self.scores[self.player1.name] > self.scores[self.player2.name]:
            return self.player1.name
        elif self.scores[self.player1.name] < self.scores[self.player2.name]:
            return self.player2.name
        else:
            return "tie"
