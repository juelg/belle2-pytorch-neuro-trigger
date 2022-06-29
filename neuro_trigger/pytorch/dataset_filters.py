from typing import List
from abc import ABC, abstractmethod

import torch



# Ideas: dataset filters not via functions but via boolean vector: which is included which is not
# -> in that case experts can also be just a filter
# factory: the dataset should return datasets itself, so like a meta object or meta class

# think about datasets which do not completly fit into ram -> own dataset that loads for each request? or batch, but we dont know the batches


# helper functions
def index2mask_array(index_array, n):
    mask_array = torch.zeros(n, dtype=int)
    mask_array[index_array] = 1
    return mask_array.bool()




class Filter(ABC):
    # use torch.where to get the keep indexes for the bool arrays
    @abstractmethod
    def fltr(self, data: torch.Tensor):
        # return a boolean vector where the entry is set to true if you want to keep the entry, otherwise to false
        pass

class ConCatFilter(Filter):
    def __init__(self, filters: List[Filter]):
        Filter.__init__(self)
        self.filters = filters

    def fltr(self, data):
        # vectors for data to keep
        keep_vecs = [filter.fltr(data) for filter in self.filters]
        # AND the keep vectors
        out = torch.ones(len(data), dtype=torch.bool)
        for vec in keep_vecs:
            out &= vec
        return out

class IdenityFilter(Filter):
    def fltr(self, data):
        return torch.ones(len(data)).bool()


class ExpertFilter(Filter):
    def __init__(self, expert: int):
        Filter.__init__(self)
        self.expert = expert
    def fltr(self, data):
        if self.expert == -1:
            # no expert filter should be applied in that case
            return torch.ones(len(data)).bool()
        return data["expert"] == self.expert


class Max2EventsFilter(Filter):
    def fltr(self, data):
        return data["ntracks"] > 2

class DuplicateEventsFilter(Filter):
    def fltr(self, data):
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

        return index2mask_array(keep_idx, len(data))