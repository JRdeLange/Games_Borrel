INSTALLATION INSTRUCTIONS

Run `pyenv local 3.11.5` (or another 3.11.x)

Run `poetry env use 3.11.5` (or another 3.11.x)

Run `poetry install`

RULES

- Internet and ChatGPT are free game.
- Cheating by walking over to another player and looking at their code is free game.
- You are only allowed to push your own `name`.py.
- You are not allowed to create any new files from within your `name`.py.
- You are not allowed to store any information outside of your `name` class.
- You are respectfully asked to keep the execution time of the benchmark cells in `testing.ipynb` under 30 seconds to facilitate timely scoring.
- You are free to look at all code here. That means both the games and, once pushed, policies of other players.
- If your code crashes or causes a crash due to invalid output, I will throw you out of that specific game, so no points that that round for that game.
- Keep yourself to the libraries available (most relevant are numpy, pandas, scipy and matplotlib (for some visualization during development)).

TIPS

- I expect time to be quite limited, so divide your time wisely between the games.
- You can store persistent-between-turns data in your class.
- `testing.ipynb` is set up to be used during development. It allows you to test your code against a random dummy player, as well as against other real players once they have pushed their policies.
- After the first round you can pull and run `play.ipynb` to see how your code performs against the other players' policies from the previous round.
- Ask me if you have any questions!
