# BEST students Challenge

This repository was created for the **BEST** students visiting the [Vrije Universiteit Brussel](https://www.vub.ac.be/). It contains a simplified Mario-like game, that is easily extensible, as well as an implementation of the [DQN](https://www.nature.com/articles/nature14236) algorithm, that can learn to play the game.

## Installation

- The code uses Python **3.7+**, if you don't have it on your machine, you can easily install it locally using [Miniconda](https://docs.conda.io/en/latest/miniconda.html).

- Create your `conda` environment, and activate it:

```bash
conda create -n py37 python=3.7
source activate py37
```

- Install all preliminary requirements:

```bash
pip install --user -r requirements.txt
```

- Install [PyTorch](https://pytorch.org/) for CPU only (for Python3.7):

```bash
pip install --user https://download.pytorch.org/whl/cpu/torch-1.1.0-cp37-cp37m-linux_x86_64.whl
```

## Run the code

Navigate to the project directory, and run:
```bash
python env/game.py
```
to play the game. You control the player using the `left` and `right` arrow keys, and jump with `space`.

To train the agent (this might take a while), run:
```bash
python main.py
```
The agent is saving logs in the `runs/` directory, to look at the logs and follow the training progress, run:
```bash
tensorboard --logdir runs/
```
This will create a dashboard that you can see in your browser.
Additionally, the agent regularly creates checkpoints of itself, also located in the `runs` directory. If you want to run an already trained agent, simply run:
```bash
python main.py --load PATH_TO_CHECKPOINT.tar
```

## Code structure

The code is structured as follows:
```
.
│   main.py (train the agent to play the game)
│
└───env
│   │   game.py (all the code for the game)
│   │   env.py (a wrapper to interact with the Reinforcement Learning agent)
│   │   level.csv (the level that will be loaded in the game)
│   │
│   └───img (a folder with the game assets)
│
└───rl
│   │   agent.py (a simple loop to interact with the game) 
│   │   dqn.py (the learning algorithm)
│   │   estimator.py (a wrapper around the Q approximator)
```

### Make your level
You can easily make your own level by looking at `level.csv`, which shows a 2D grid representing the level structure:

- `0` stands for nothing (air)
- `1` stands for the ground
- `2` stands for the goal position, reaching it means you win the level
- `3` stands for a walker enemy (an enemy that walks to the left once it sees the player)

The player always starts at position `(2,8)`, so that spot needs to be free.

### Extend the game
The game has a very simple structure:
```
GameObject: Every element of the game (such as the Player or the Ground) is a GameObject
│
└───Ground: A singleton class representing all the ground tiles
│
└───Goal: A class with the goal position
│
└───Moveable: All elements that can move around. 
              It includes basic movement (`update_speed`),
              and collision detection with the ground (`handle_obstacle_collision`)
              - A Moveable has an `update` method, that handle the object's logic at each timestep
│   │
│   └───Player: The player class. Has logic to handle collision with enemies (kills when on top)and goal-position
│   │
│   └───Enemy: All enemies inherit from this class.
               It handles player collision (dies or kills the player),
               and other enemies collision (simply bump into each other)
│   │   │
│   │   └───Walker: The only implemented enemy of the game.
                    Moves to the right when it sees the player
```

Every `GameObject` has an associated `GameSprite`, with the image representing the object.

The main class is `Game`. The method that does most of the work is `update`: it updates every element of the game and shifts their position so that they stay visible on the screen.
This is then called in the `step` method, which draws the current frame on the screen, and checks if the game terminates.

---

It should be pretty straightforward to add elements like additional enemies, moving platforms, falling spikes...
