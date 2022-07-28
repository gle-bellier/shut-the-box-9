from typing import Tuple, List
import numpy as np
from src.game.rules import combinaisons


class Board():

    def __init__(self):
        """Initializes board.
        """

        # the first row is the up tiles and the second one the downs

        self.board = np.zeros((9, 2))
        self.board[:, 0] += 1

    def get_board(self) -> np.ndarray:
        """Returns the board.

        Returns:
            np.ndarray: board.
        """
        return self.board

    def check_under_six(self) -> bool:
        """Checks if all tile stricly above 6 are down. 

        Returns:
            bool: result of the test.
        """

        return np.sum(self.board[6:, 0]) == 0

    def check_end(self) -> bool:
        """Tests if all tiles are down.

        Returns:
            bool: True if all tiles are down.
        """

        return not np.any(self.board[:, 0])

    def check_action(self, action: Tuple[int]) -> bool:
        """Checks if the action is valid or not

        Args:
            action (Tuple[int]): tile indexes to shut down.

        Returns:
            bool: True if the action is valid, else False.
        """
        for index in action:

            if index > len(self.board) - 1 or index < 0:
                return False

            elif not self.board[index, 0]:
                # the tile is already down
                return False

        return True

    def get_actions(self, dice_sum: int) -> List[Tuple[int]]:
        """Computes all combinaison of tiles index such that the tile sum equal the 
        dice sum.

        Args:
            dice_sum (int): sum of the dice.

        Returns:
            List[Tuple[int]]: list of actions.
        """
        return combinaisons[dice_sum]

    def get_legal_actions(self, dice_sum: int) -> List[Tuple[int]]:
        """Returns legal combinaison of tiles index such that the tile sum equal the 
        dice sum. Returns [] if any. 

        Args:
            dice_sum (int): sum of the dice.

        Returns:
            List[Tuple[int]]: list of legal actions.
        """
        return [
            action for action in combinaisons[dice_sum]
            if self.check_action(action)
        ]

    def shut_down(self, action: Tuple[int]) -> None:
        """Shuts the tiles down.

        Args:
            action (Tuple[int]): indexes of the tiles to shut
            down.
        """
        for index in action:
            self.board[index, 0] = 0
            self.board[index, 1] = 1

    def play_action(self, action: Tuple[int]) -> bool:
        """Plays one move.

        Args:
            action (Tuple[int]): tile indexes to shut down.

        Returns:
            bool: True if the move is done, else False
        """
        if self.check_action(action):
            self.shut_down(action)
            return True
        else:
            return False

    def __rows2str(self, row: np.ndarray) -> str:
        """Computes string corresponding to one row of the board.s

        Args:
            row (np.ndarray): row of shape (L, 1), 0 when tile down
            1 when up.

        Returns:
            str: string representing the row.
        """
        digits = np.arange(1, 10, 1)
        return "|".join([f"{i:.0f}" if i else "X" for i in row * digits])

    def __repr__(self) -> str:
        """Returns representation string of the board.

        Returns:
            str: board string.
        """

        ups = self.__rows2str(self.board[:, 0])
        downs = self.__rows2str(self.board[:, 1])
        return ups + "\n" + downs
