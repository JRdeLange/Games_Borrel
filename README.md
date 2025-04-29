WHAT THE EVEN?

Hi! This is my agent/policy programming borrel where you will fight to the death, with your code as your champion!

You will have until 16:00 to push your code. At that point I will run `round1.ipynb` to obtain the first round of points.

Next, you will have until 18:00 to push your code. At that point I will run `round2.ipynb` to obtain the second round of points, which counts double.

You should implement the functions in `borrel/policies/name.py` (where `name` is your name) to be able to play the games.

For ease of use you can just push to main, but do make very sure you do not push anything else than your `name.py` file. Otherwise, you get points deducted!

INSTALLATION INSTRUCTIONS

Run `pyenv install 3.11.5` (or another 3.11.x)

Run `pyenv local 3.11.5` (or another 3.11.x)

Run `poetry env use 3.11.5` (or another 3.11.x)

Run `poetry install`

RULES

- Internet and ChatGPT are free game.
- Cheating by walking over to another player and looking at their code is free game.
- You are only allowed to push your own `name`.py.
- You are not allowed to create any new files from within your `name`.py.
- You are not allowed to store any information outside of your `name` class.
- You are respectfully asked to keep the execution time of the benchmark cells in `testing.ipynb` under 45 seconds to facilitate timely scoring.
- You are free to look at all code here. That means both the games and, once pushed by them and pulled by you, policies of other players.
- If your code crashes or causes a crash due to invalid output, I will throw you out of that specific game, so no points that that round for that game.
- Keep yourself to the libraries available (most relevant are numpy, pandas, scipy and matplotlib (for some visualization during development)).
- If you discover a significant (up to my judgement) bug, let me know and get a bonus point!

TIPS

- I expect time to be quite limited, so divide your time wisely between the games.
- You can store persistent-between-turns data in your class.
- `testing.ipynb` is set up to be used during development. It allows you to test your code against a random dummy player, as well as against other real players once they have pushed their policies.
- After the first round you can pull and run `play.ipynb` to see how your code performs against the other players' policies from the previous round.
- Ask me if you have any questions!
