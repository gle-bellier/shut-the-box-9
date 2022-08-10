# Shut the box 9


This project aims at using reinforcement learning techniques to master the game named "Shut the box 9". ([Rules](https://en.wikipedia.org/wiki/Shut_the_box)).

The idea here is to consider this game as a great exercice to implement reinforcement learning algorithms and to learn to train models with the _Jax_ framework (with the _Flax_ library for the neural networks). 

## Project structure

We only propose algorithms that do not rely on humain annotated games or humain played games. All the data pipeline can be found in the `/src/data` folder.

The game in itself is implemented in the `/src/game` folder and a memory buffer of the played moves is added to the scripts.

The trained (or random) agents are available in `/src/agents`. All the scripts dealing with agents training and neural networks architectures are in `/src/learn`.

Some tests are also implemented to ensure the robustness of the training pipeline in the `/tests` folder

## Installation

1. Clone this repository:

```bash
git clone git@github.com:gle-bellier/stb9.git

```

2. Install requirements:

```bash
cd stb9
pip install -r requirements.txt

```