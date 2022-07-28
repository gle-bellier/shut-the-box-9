from typing import Tuple
import numpy as np

from src.agents.agent import Agent
from src.game.board import Board


class RandomAgent(Agent):

    def __init__(self):
        super().__init__()

    def select_action(self, board: Board, dice_sum: int) -> Tuple[int]:
        """Randomly select the action to play.

        Args:
            board (Board): board of the game.
            dice_sum (int): sum of the dice.

        Returns:
            Tuple[int]: action selected.
        """

        # select a random action

        actions = board.get_legal_actions(dice_sum)
        i_action = np.random.randint(0, len(actions))

        return actions[i_action]
