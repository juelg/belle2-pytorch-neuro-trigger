import logging
import unittest
import sys

import torch
from neuro_trigger import main
from neuro_trigger.pytorch.dataset import BelleIIDataManager
from neuro_trigger.pytorch.dataset_filters import ConCatFilter, DuplicateEventsFilter, IdenityFilter, Max2EventsFilter, index2mask_array
sys.path.append("/mnt/scratch/juelg/neuro-trigger-v2")

from neuro_trigger.main import DATA_DEBUG, create_trainer_pl_module, prepare_vars


# TODO
# need a subset of training data in this folder
# run fast dev run and overfit batches




class End2End(unittest.TestCase):
    TEST_DATA = "neuro_trigger/tests/test_data_filter.csv"

    def test_end2end_fast_dev_run(self):
        used_config = "base"
        data = (self.TEST_DATA, self.TEST_DATA, self.TEST_DATA)

        hparams, log_folder, experts, version, experts_str, logger = prepare_vars(used_config, debug=True)

        trainers_modules = [create_trainer_pl_module(expert_i, experts, log_folder, hparams, data, version, fast_dev_run=True) for expert_i in range(len(experts))]


        trainers_modules[0][0].fit(trainers_modules[0][1])


    def test_end2end_overfit(self):
        used_config = "base"
        data = (self.TEST_DATA, self.TEST_DATA, self.TEST_DATA)

        hparams, log_folder, experts, version, experts_str, logger = prepare_vars(used_config, debug=True)

        trainers_modules = [create_trainer_pl_module(expert_i, experts, log_folder, hparams, data, version, overfit_batches=1) for expert_i in range(len(experts))]
        trainers_modules[0][0].fit(trainers_modules[0][1])


    # complete blackbox training with 2 epochs and check if all files exist
    def test_blackbox(self):
        data = (self.TEST_DATA, self.TEST_DATA, self.TEST_DATA)
        log_folder = main.main(config="baseline_v2", data=data, debug=True, solo_expert=False)
        # we should have summery.json, weights.json, log from all experts, prediction_random1
        #         app.log       expert_1      expert_2.log  expert_4      git_id.txt             pred_data_random3.csv  prediction_random3.pt
        # expert_0      expert_1.log  expert_3      expert_4.log  pred_data_random1.csv  prediction_random1.pt  summary.json
        # expert_0.log  expert_2      expert_3.log  git_diff.txt  pred_data_random2.csv  prediction_random2.pt  weights.json

    def test_blackbox_solo_expert(self):
        # expert=-1 -> only one expert
        data = (self.TEST_DATA, self.TEST_DATA, self.TEST_DATA)
        main.main(config="baseline_v2", data=data, debug=True, solo_expert=True)
        


# TODO: tests for distribution
class FilterTest(unittest.TestCase):
    TEST_DATA = "neuro_trigger/tests/test_data_filter.csv"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dm = BelleIIDataManager(self.TEST_DATA, logging.getLogger())

    def test_index2mask_array(self):
        index_array = torch.tensor([1, 3, 8, 10])
        n = 11
        b_array = index2mask_array(index_array, n)
        self.assertTrue((b_array == torch.tensor([False, True, False, True, False, False, False, False, True, False, True])).all())

    def test_identityfilter(self):
        d = self.dm.dataset(filter=IdenityFilter())
        le = 51
        self.assertEqual(le, len(d))

    def test_expertfilter(self):
        d = self.dm.expert_dataset(expert=0)
        exp_0 = 38
        self.assertEqual(exp_0, len(d))

        d = self.dm.expert_dataset(expert=4)
        exp_4 = 4
        self.assertEqual(exp_4, len(d))

    def test_Max2EventsFilter(self):
        # TODO: filter on reco tracks
        d = self.dm.dataset(filter=Max2EventsFilter())
        le = 34
        self.assertEqual(le, len(d))

    def test_DuplicateEventsFilter(self):
        d = self.dm.dataset(filter=DuplicateEventsFilter())
        le = 24
        self.assertEqual(le, len(d))
        et = [(event, track) for event, track in zip(d.data["event"], d.data["track"])]
        # event 13824, track 0 should be in it
        self.assertTrue((13824, 0) in et)
        # event 13824, track 1 should not be in it
        self.assertTrue((13824, 1) not in et)

    def test_ConCatFilter(self):
        d = self.dm.dataset(filter=ConCatFilter(filters=[DuplicateEventsFilter(), Max2EventsFilter(), IdenityFilter()]))
        le = 15
        self.assertEqual(le, len(d))



if __name__ == '__main__':
    unittest.main()
