import pickle

from src.agents.random_agent import RandomAgent
from src.data.builders.dataset_builder import GameDatasetBuilder

from effortless_config import Config


class config(Config):
    NB_GAMES = 100
    FILENAME = "games_resumes.pickle"


config.parse_args()
gdb = GameDatasetBuilder(config.NB_GAMES, "data/games/" + config.FILENAME,
                         RandomAgent, dict())
gdb.build()
