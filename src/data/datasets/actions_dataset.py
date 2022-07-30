import numpy as np
from typing import List

from src.data.utils import read_pickle


class ActionsDataset:

    def __init__(self, actions: List[dict]) -> None:
        """Initialize the ActionsDataset.

        Args:
            actions (List[dict]): list of all actions.
        """
        self.actions = actions

    def __len__(self) -> int:
        """Return length of the dataset.

        Returns:
            int: length of the dataset.
        """
        return len(self.actions)

    def __getitem__(self, i: int) -> List[np.ndarray or int]:
        """Return the ith action of the dataset.

        Args:
            i (int): index of the action.

        Returns:
            List[np.ndarray or int]: in this order: board, 
            sum of di(c)e, action, score. 
        """

        i = max(0, min(len(self) - 1, i))

        return list(self.actions[i].values())