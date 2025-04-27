import copy

import pandas as pd


class Minority:
    def __init__(self, players: list, render: bool = False):
        self.players = players
        self.render = render

        self.scores = {player.name: 0 for player in players}

        names = [player.name for player in players]
        assert len(names) == len(set(names)), "Player names must be unique."
        self.history = pd.DataFrame(
            columns=[player.name for player in players] + ["rounds_remaining"]
        )

    def play_round(self, rounds_remaining: int):
        round = {}
        for player in self.players:
            choice = player.minority(copy.deepcopy(self.history))
            round[player.name] = choice
        round["rounds_remaining"] = rounds_remaining

        if self.render:
            print(round)

        As = sum([1 for choice in round.values() if choice == "A"])
        Bs = sum([1 for choice in round.values() if choice == "B"])

        for player in self.players:
            if round[player.name] == "A" and As < Bs:
                self.scores[player.name] += 1
            elif round[player.name] == "B" and Bs < As:
                self.scores[player.name] += 1

        self.history = pd.concat(
            [self.history, pd.DataFrame([round])], ignore_index=True
        )

    def get_history(self):
        return self.history

    def get_scores(self):
        return self.scores

    def reset(self):
        self.scores = {player.name: 0 for player in self.players}
        self.history = pd.DataFrame(columns=[player.name for player in self.players])

    def get_winner(self):
        max_score = max(self.scores.values())
        winners = [name for name, score in self.scores.items() if score == max_score]

        return winners

    def play_game(self, rounds: int = 100):
        self.reset()

        for i in range(rounds):
            self.play_round(rounds - i - 1)

        return self.get_winner()
