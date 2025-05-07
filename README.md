Simple example using Q-learning, a foundational RL algorithm

Vibe-coded with OpenAI.

- [Example: Teaching an Agent to Reach a Goal in a Grid](#example-teaching-an-agent-to-reach-a-goal-in-a-grid)
- [Installation](#installation)
- [Run the animation](#run-the-animation)

# Example: Teaching an Agent to Reach a Goal in a Grid

The environment is a 4x4 grid where:

* The agent starts in a random position.
* There’s a fixed goal cell that gives a reward.
* Every step has a small penalty to encourage fast learning.
* The agent can move: UP, DOWN, LEFT, RIGHT.
```
S . . .
. . . .
. . . .
. . . G
```
* S = Start (random)
* G = Goal (reward = +10)
* Each move = -1 reward
* Goal ends the episode

> Note: diagonal moves are not used.
> * Simplicity: Fewer actions mean a smaller Q-table and simpler policy learning.
> * Clarity: Easier to visualize and debug step-by-step movement.
> * Tradition: Many tutorials and textbooks use 4-action agents for foundational learning.

# Installation

Create a virtual environment and install packages.  Tested on Ubuntu 22.04.

```console
sudo apt install -y python3.10-venv
sudo apt-get install python3-tk

cd
python3 -m venv venv_rl
source ./venv_rl/bin/activate

cd rl

pip install --upgrade pip
pip install -r requirements.txt
```

# Run the animation
```console
cd
source ./venv_rl/bin/activate

cd rl
python3 main.py
```

Example run.  The blue square is the goal.

![](assets/q_learning_animation.gif)
