from abc import ABC, abstractmethod
from typing import Dict, Iterable, List

import numpy as np
import torch


# helper function
def index2mask_array(index_array: torch.Tensor, n: int) -> torch.Tensor:
    """Helper function which creates boolean array out of an index array

    Transforms an array that describes a subset of array A by the indexes that are in the subset
    to bit mask array, which is set to true if the respective index should be kept and false otherwise

    Args:
        index_array (torch.Tensor): index array (1d tensor), array with indexes that should be kept
        n (int): length of the original array

    Returns:
        torch.Tensor: boolean array with entries set to true if element of index_array, else false
    """
    mask_array = torch.zeros(n, dtype=int)
    mask_array[index_array.long()] = 1
    return mask_array.bool()


class Filter(ABC):
    """Abstract dataset filter

    Must implement the fltr function which takes the dataset and returns a boolean array with entries
    set to true that should be kept.
    """

    @abstractmethod
    def fltr(self, data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Should implement a concrete filter: Can calculate which elements to keep given the dataset

        Args:
            data (torch.Tensor): Dataset data

        Returns:
            torch.Tensor: Returns a boolean vector where the entry is set to true if you want to keep the entry, otherwise to false
        """
        pass


class ConCatFilter(Filter):
    """Concatenates several filters given as a list by using the "and" operator"""

    def __init__(self, filters: List[Filter]):
        """
        Args:
            filters (List[Filter]): list of filters that should be concatenated
        """
        Filter.__init__(self)
        self.filters = filters

    def fltr(self, data: Dict[str, torch.Tensor]) -> torch.Tensor:
        # vectors for data to keep
        keep_vecs = [filter.fltr(data) for filter in self.filters]
        # AND the keep vectors
        out = torch.ones(len(data["x"]), dtype=torch.bool)
        for vec in keep_vecs:
            out &= vec
        return out


class IdentityFilter(Filter):
    """No filter, returns array with all values set to true"""

    def fltr(self, data: torch.Tensor) -> torch.Tensor:
        return torch.ones(len(data["x"])).bool()


class ExpertFilter(Filter):
    def __init__(self, expert: int):
        """Filters out data which does not belong to the given expert
        Args:
            expert (int): expert number in {0, 1, 2, 3, 4}
        """
        Filter.__init__(self)
        self.expert = expert

    def fltr(self, data: Dict[str, torch.Tensor]) -> torch.Tensor:
        if self.expert == -1:
            # no expert filter should be applied in that case
            return torch.ones(len(data["x"])).bool()
        return data["expert"] == self.expert


class Max2EventsFilter(Filter):
    """Keeps only events with less than 3 tracks"""

    def fltr(self, data: Dict[str, torch.Tensor]) -> torch.Tensor:
        # create map event -> y -> tracks
        event_map = {}
        for idx, (e, y) in enumerate(zip(data["event"], data["y"])):
            ey = f"{e}"
            k = f"{y[0]},{y[1]}"
            if ey not in event_map:
                event_map[ey] = {}
            if k not in event_map[ey]:
                event_map[ey][k] = []
            event_map[ey][k].append(idx)
        keep_idx = []
        for event, value in event_map.items():
            if len(value) <= 2:
                for key, value in event_map[event].items():
                    keep_idx += value
                # keep_events.append(event)

        keep_idx = torch.Tensor(keep_idx)
        return index2mask_array(keep_idx, len(data["x"]))


class DuplicateEventsFilter(Filter):
    """If there is several data for a single track, only the first one is kept and the others are filtered out"""

    def fltr(self, data: Dict[str, torch.Tensor]) -> torch.Tensor:
        # create map event,y -> tracks
        event_map = {}
        for idx, (e, y) in enumerate(zip(data["event"], data["y"])):
            ey = f"{e},{y[0]},{y[1]}"
            if ey not in event_map:
                event_map[ey] = []
            event_map[ey].append(idx)
        keep_idx = []
        for key, value in event_map.items():
            keep_idx.append(value[0])
        keep_idx = torch.Tensor(keep_idx)

        return index2mask_array(keep_idx, len(data["x"]))


class RangeFilter(Filter):
    def __init__(self, rng: Iterable):
        """Given out an index range, this filter removes all non included indexes from the dataset
        Args:
            rng (Iterable): Something that describes a range, e.g. range(1, 100)
        """
        self.rng = torch.tensor(np.array([i for i in rng]))

    def fltr(self, data: Dict[str, torch.Tensor]) -> torch.Tensor:
        assert len(self.rng) <= len(data["x"])
        assert min(self.rng) >= 0
        assert max(self.rng) < len(data["x"])
        return index2mask_array(self.rng, len(data["x"]))
