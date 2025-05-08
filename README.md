Simple example using Q-learning, a foundational reinforcement learning (RL) algorithm.

Vibe-coded with OpenAI's ChatGPT.

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

> Note: diagonal moves are not allowed.
> * Simplicity: Fewer actions mean a smaller Q-table and simpler policy learning.
> * Clarity: Easier to visualize and debug step-by-step movement.
> * Tradition: Many tutorials and textbooks use 4-action agents for learning.

# Installation

Tested on Ubuntu Desktop 22.04.

First clone this repo.
```console
sudo apt install git git-lfs

git clone git@github.com:guynich/rl_q_learning.git
```

Create a virtual environment and install packages.
```console
sudo apt install -y python3.10-venv
sudo apt-get install python3-tk

cd
python3 -m venv venv_rl
source ./venv_rl/bin/activate

cd rl_q_learning

pip install --upgrade pip
pip install -r requirements.txt
```

# Run the animation
```console
cd
source ./venv_rl/bin/activate

cd rl_q_learning
python3 main.py
```

Example run with 100 episodes.  The blue square is the goal.

![](assets/q_learning_animation.gif)

Reload this web page to restart the animation.

# Next steps

Great demo for Deep Q Learning : https://projects.rajivshah.com/rldemo/

