from random import random
from src.agents.random_agent import RandomAgent
from src.game.board import Board
from src.game.game import Game

n_simulations = 10_000

random_agent = RandomAgent()
mean_score = 0

for _ in range(n_simulations):
    board = Board()
    game = Game(board, random_agent)

    score = game.run_game()
    mean_score += (score) / n_simulations

print(mean_score)