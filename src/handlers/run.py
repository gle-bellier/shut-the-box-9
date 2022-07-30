from random import random
from src.agents.random_agent import RandomAgent
from src.game.board import Board
from src.game.game import Game

board = Board()
random_agent = RandomAgent()

game = Game(board, random_agent)

done = game.run_game()
print(game.buffer.get_resume())