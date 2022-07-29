from typing import Tuple, List
from src.game.board import Board


class Buffer:

    def __init__(self, board: Board) -> None:
        """Initialize buffer.

        Args:
            board (Board): initial board.
        """
        self.board = board

        self.steps = []
        self.score = None

    def add_step(self, sum_dice: int, action: Tuple[int]) -> None:
        """Add step to the buffer of steps.

        Args:
            sum_dice (int): sum of the di(c)e.
            action (Tuple[int]): action played by the agent.
        """
        self.steps.append({"sum_dice": sum_dice, "action": action})

    def set_end_score(self, score: int) -> None:
        """Set the end score of the game.

        Args:
            score (int): end score.
        """
        self.score = score
