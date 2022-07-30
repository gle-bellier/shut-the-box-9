import pickle
from tqdm import tqdm
from typing import List, Tuple, Type

from src.agents.random_agent import RandomAgent
from src.agents.agent import Agent
from src.game.board import Board
from src.game.game import Game
from src.data.utils import add_to_pickle


class GameDatasetBuilder:

    def __init__(self, num_games: int, save_path: str, agent: Type[Agent],
                 agent_kwargs: dict) -> None:
        """Initialize GameDatasetBuilder

        Args:
            num_games (int): number of games to emulate.
            save_path (str): path to the saving file.
            agent (Type[Agent]): type of agent to use to simulate.
            agent_kwargs (dict): kwargs of the agent used.
        """

        self.num_games = num_games
        self.agent = agent
        self.agent_kwargs = agent_kwargs
        self.save_path = save_path

    def build(self, save=True) -> None:
        """Build the dataset.

        Args:
            save (bool, optional): if True then the game actions
            are saved. Defaults to True.
        """

        for _ in tqdm(range(self.num_games)):
            # run a game
            agent = self.agent(**self.agent_kwargs)
            board = Board()

            game = Game(board, agent)
            game.run_game()

            resume = game.get_resume()
            if save:
                add_to_pickle(resume, self.save_path)
