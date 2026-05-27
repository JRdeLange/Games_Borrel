import copy
import warnings

import numpy as np
import pandas as pd

ROCK = "r"
PAPER = "p"
SCISSORS = "s"
GUN = "g"
DUCK = "d"

WHO_WINS = {
    (ROCK, ROCK): "tie",
    (PAPER, ROCK): "1",
    (SCISSORS, ROCK): "2",
    (GUN, ROCK): "%",
    (DUCK, ROCK): "1",
    #
    (ROCK, PAPER): "2",
    (PAPER, PAPER): "tie",
    (SCISSORS, PAPER): "1",
    (GUN, PAPER): "1",
    (DUCK, PAPER): "2",
    #
    (ROCK, SCISSORS): "1",
    (PAPER, SCISSORS): "2",
    (SCISSORS, SCISSORS): "tie",
    (GUN, SCISSORS): "1",
    (DUCK, SCISSORS): "2",
    #
    (ROCK, GUN): "%",
    (PAPER, GUN): "2",
    (SCISSORS, GUN): "2",
    (GUN, GUN): "%",
    (DUCK, GUN): "1",
    #
    (ROCK, DUCK): "2",
    (PAPER, DUCK): "1",
    (SCISSORS, DUCK): "1",
    (GUN, DUCK): "2",
    (DUCK, DUCK): "tie",
}

SCORE_BASE = 100

INTEREST_RATE = 0.1

MAX_LEND_AMOUNT = 2000

RELOAD_TIME = 5


class RPSPlayer:
    def __init__(self, player):
        self.name = player.name
        self.player = player
        # resources
        self.reload_timer = RELOAD_TIME
        self.n_reversals = 25
        self.score = 0

    def decrement_reload_timer(self):
        if self.reload_timer > 0:
            self.reload_timer -= 1

    def shoot(self):
        if self.reload_timer > 0:
            return False
        self.reload_timer = RELOAD_TIME
        return True

    def reverse(self):
        if self.n_reversals <= 0:
            return False
        self.n_reversals -= 1
        return True

    def is_bet_legal(self, amount):
        if not isinstance(amount, int) or amount < 0:
            return False
        if amount > self.score:
            # amount which needs to be lended
            lend_amount = amount - self.score
            if lend_amount > MAX_LEND_AMOUNT:
                return False
        return True

    def resolve_bet(self, amount, won):
        if won:
            self.score += amount
        else:
            self.score -= amount

    def pay_interest(self):
        if self.score < 0:
            interest = int(-self.score * 0.1)  # 10% interest on negative score
            self.score -= interest
            return interest


class RPS_Gun:
    def __init__(self, player1, player2, render: bool = False):
        self.player1 = RPSPlayer(player1)
        self.player2 = RPSPlayer(player2)
        self.render = render

        self.history = pd.DataFrame(
            columns=[
                player1.name,
                player2.name,
                player1.name + "_bet",
                player2.name + "_bet",
                player1.name + "_score",
                player2.name + "_score",
                player1.name + "_reload_timer",
                player2.name + "_reload_timer",
                "rounds_remaining",
            ]
        )

    def determine_winner_loser(self, player1_choice, player2_choice):
        result = WHO_WINS.get((player1_choice, player2_choice))
        if result == "1":
            return self.player1
        elif result == "2":
            return self.player2
        elif result == "%":
            return self.player1 if np.random.rand() < 0.5 else self.player2
        elif result == "tie":
            return None
        else:
            raise ValueError(f"Invalid choices: {player1_choice}, {player2_choice}")

    def play_round(self, rounds_remaining: int):
        """Returns the name of the winner, or 'tie' if it's a tie."""
        self.player1.decrement_reload_timer()
        self.player2.decrement_reload_timer()

        round_row = {
            self.player1.name + "_reload_timer": self.player1.reload_timer,
            self.player2.name + "_reload_timer": self.player2.reload_timer,
            "rounds_remaining": rounds_remaining,
        }

        player1_choice, player1_bet = self.player1.player.rps_gun(
            copy.deepcopy(self.history),
        )
        if player1_choice not in [ROCK, PAPER, SCISSORS, GUN, DUCK]:
            return self.player2.name

        player2_choice, player2_bet = self.player2.player.rps_gun(
            copy.deepcopy(self.history),
        )
        if player2_choice not in [ROCK, PAPER, SCISSORS, GUN, DUCK]:
            return self.player1.name

        round_row[self.player1.name] = player1_choice
        round_row[self.player2.name] = player2_choice

        player_1_illegal = False
        player_2_illegal = False

        if player1_choice == GUN and not self.player1.shoot():
            player_1_illegal = True
            if self.render:
                print(f"{self.player1.name} tried to shoot but is reloading!")
        if player2_choice == GUN and not self.player2.shoot():
            player_2_illegal = True
            if self.render:
                print(f"{self.player2.name} tried to shoot but is reloading!")
        if not self.player1.is_bet_legal(player1_bet):
            player_1_illegal = True
            if self.render:
                print(f"{self.player1.name} tried to bet illegally!")
        if not self.player2.is_bet_legal(player2_bet):
            player_2_illegal = True
            if self.render:
                print(f"{self.player2.name} tried to bet illegally!")

        if player_1_illegal and player_2_illegal:
            winner = None
        elif player_1_illegal:
            winner = self.player2
        elif player_2_illegal:
            winner = self.player1
        else:
            winner = self.determine_winner_loser(player1_choice, player2_choice)

        if winner is not None:
            winner.score += SCORE_BASE

        if winner == self.player1:
            self.player1.resolve_bet(player1_bet, True)
            self.player2.resolve_bet(player2_bet, False)

        elif winner == self.player2:
            self.player1.resolve_bet(player1_bet, False)
            self.player2.resolve_bet(player2_bet, True)

        round_row[self.player1.name + "_bet"] = player1_bet
        round_row[self.player2.name + "_bet"] = player2_bet
        round_row[self.player1.name + "_score"] = self.player1.score
        round_row[self.player2.name + "_score"] = self.player2.score

        self.player1.pay_interest()
        self.player2.pay_interest()

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
                f"{self.player1.name} bet: {player1_bet}, {self.player2.name} bet: {player2_bet}"
            )
            print(
                f"{self.player1.name} score: {self.player1.score}, {self.player2.name} score: {self.player2.score}"
            )
            print(
                f"{self.player1.name} reload timer: {self.player1.reload_timer}, {self.player2.name} reload timer: {self.player2.reload_timer}"
            )
            print(
                f"Round result: {winner.name if winner else 'tie'}"
                + (f" gets {SCORE_BASE} points" if winner else "")
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
                self.player1.name + "_reload_timer",
                self.player2.name + "_reload_timer",
                "rounds_remaining",
            ]
        )

    def play_game(self, rounds: int = 400):
        self.reset()

        for i in range(rounds):
            self.play_round(rounds - i - 1)

        if self.player1.score > self.player2.score:
            return self.player1.name
        elif self.player1.score < self.player2.score:
            return self.player2.name
        else:
            return "tie"
