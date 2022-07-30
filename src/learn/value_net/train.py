import jax.numpy as jnp

from src.data.datasets.actions_dataloader_generator import ActionsDatasetGenerator
from src.data.datasets.embedder import Embedder

adg = ActionsDatasetGenerator("data/games/games_resumes.pickle", 3, Embedder())
train, val, test = adg.split()

for batch in train:
    board, sum_dice, score = batch
    print(board.shape)
    print(sum_dice.shape)
    print(score.shape)