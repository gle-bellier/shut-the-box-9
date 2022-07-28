from typing import Tuple, List
from abc import ABC, abstractmethod

from src.game.board import Board


class Agent(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def select_action(self, board: Board, sum_dice: int) -> Tuple[int]:
        """Return the action to play. This action is always legal.
        """
        pass