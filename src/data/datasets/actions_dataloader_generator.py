from typing import List, Tuple
import random
from src.data.datasets.embedder import Embedder

from src.data.datasets.np_dataloader import NumpyLoader
from src.data.utils import read_pickle
from src.data.datasets.actions_dataset import ActionsDataset


class ActionsDatasetGenerator:
    def __init__(self,
                 path: str,
                 batch_size: int,
                 embedder: Embedder,
                 nb_actions_max=None) -> None:
        """Initialize the reader.

        Args:
            path (str): path to the pickle file.
            batch_size (int): batch size.
            embedder (Embedder): data embedder.
            nb_actions_max (int, optional): maximum number of actions 
            to return. Defaults to None, i.e. returning all actions from 
            the dataset.
        """
        self.path = path
        self.batch_size = batch_size
        self.embedder = embedder
        self.actions = self.load_actions(nb_actions_max)

    def load_actions(self, nb_actions_max=None) -> List[dict]:
        """Read actions from the pickle dataset file.

        Args:
            nb_actions_max (int, optional): maximum number of actions 
            to return. Defaults to None, i.e. returning all actions from 
            the dataset.

        Returns:
            List[dict]: List of actions.
        """

        actions = []
        actions_gen = read_pickle(self.path)

        if nb_actions_max is None:
            actions = [action for action in actions_gen]
        else:
            try:
                while len(actions) < nb_actions_max:
                    actions.append(next(actions_gen))

            except:
                pass

        return actions

    def __split_indexes(self, N: int, prop: List[float]) -> List[Tuple[int]]:
        """Compute the cut indexes for the three datasets with respect
        to the proportions.

        Args:
            N (int): length of the full dataset.
            prop (List[float]): proportions.

        Returns:
            List[Tuple[int]]: list of (start, end) indexes.
        """

        train_start, train_end = 0, int(N * prop[0])
        val_start, val_end = train_end, train_end + int(N * prop[1])
        test_start, test_end = val_end, val_end + int(N * prop[2])

        return [(train_start, train_end), (val_start, val_end),
                (test_start, test_end)]

    def split(self,
              proportions=[0.8, 0.1, 0.1],
              shuffle=True) -> Tuple[ActionsDataset]:
        """Split the dataset into train, val, test dataloaders.

        Args:
            proportions (list, optional): proportions of the datasets.
            Defaults to [0.8, 0.1, 0.1].
            shuffle (bool, optional): shuffling if True. Defaults to True.

        Returns:
            Tuple[ActionsDataset]: train, val, test dataloaders.
        """

        if shuffle:
            random.shuffle(self.actions)

        # split full dataset in 3 wrt the proportions
        split_indexes = self.__split_indexes(len(self.actions), proportions)

        dataloaders = []
        for indexes in split_indexes:

            data = self.actions[slice(*indexes)]
            dataset = ActionsDataset(data, self.embedder)
            dataloaders += [NumpyLoader(dataset, batch_size=self.batch_size)]

        return dataloaders
