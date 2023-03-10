"""
 Copyright (c) 2021-2023 Tobias Juelg

 This program is free software: you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation, either version 3 of the License, or
 (at your option) any later version.

 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.

 You should have received a copy of the GNU General Public License
 along with this program.  If not, see <https://www.gnu.org/licenses/>.
 """

import os
import sys
import unittest
from functools import partial

from neuro_trigger.lightning.mean_tb_logger import MeanTBLogger

sys.path.append("/mnt/scratch/juelg/neuro-trigger-v2")
import numpy as np
import torch
from scipy.stats import norm, uniform

from neuro_trigger import main, utils
from neuro_trigger.main import DATA_DEBUG, create_trainer_pl_module, prepare_vars
from neuro_trigger.pytorch.dataset import BelleIIDataManager, BelleIIDistDataset
from neuro_trigger.pytorch.dataset_filters import (
    ConCatFilter,
    DuplicateEventsFilter,
    IdentityFilter,
    Max2EventsFilter,
    index2mask_array,
)


class End2End(unittest.TestCase):
    TEST_DATA = ["neuro_trigger/tests/test_data_filter.csv"]

    def test_end2end_fast_dev_run(self):
        used_config = "base"
        data = (self.TEST_DATA, self.TEST_DATA, self.TEST_DATA)

        hparams, log_folder, experts, version, experts_str, logger = prepare_vars(
            used_config, debug=True
        )

        mean_tb_logger = MeanTBLogger(os.path.join(log_folder, "mean_expert"), experts)
        mean_tb_logger.start_thread()

        compare_to = utils.get_compare_to_path(hparams)
        data_mgrs = [
            BelleIIDataManager(
                data[i], out_dim=hparams.out_size, compare_to=compare_to[i]
            )
            for i in range(3)
        ]

        trainers_modules = [
            create_trainer_pl_module(
                expert_i,
                experts,
                log_folder,
                hparams,
                data_mgrs,
                version,
                mean_tb_logger,
                fast_dev_run=True,
            )
            for expert_i in range(len(experts))
        ]

        trainers_modules[0][0].fit(trainers_modules[0][1])

        mean_tb_logger.stop_thread()

    def test_end2end_overfit(self):
        used_config = "base"
        data = (self.TEST_DATA, self.TEST_DATA, self.TEST_DATA)

        hparams, log_folder, experts, version, experts_str, logger = prepare_vars(
            used_config, debug=True
        )

        mean_tb_logger = MeanTBLogger(os.path.join(log_folder, "mean_expert"), experts)
        mean_tb_logger.start_thread()

        compare_to = utils.get_compare_to_path(hparams)
        data_mgrs = [
            BelleIIDataManager(
                data[i], out_dim=hparams.out_size, compare_to=compare_to[i]
            )
            for i in range(3)
        ]

        trainers_modules = [
            create_trainer_pl_module(
                expert_i,
                experts,
                log_folder,
                hparams,
                data_mgrs,
                version,
                mean_tb_logger,
                overfit_batches=1,
            )
            for expert_i in range(len(experts))
        ]
        trainers_modules[0][0].fit(trainers_modules[0][1])

        mean_tb_logger.stop_thread()

    # complete blackbox training with 2 epochs and check if all files exist
    def test_blackbox(self):
        data = (self.TEST_DATA, self.TEST_DATA, self.TEST_DATA)
        log_folder = main.main(
            config="baseline_v2", data=data, debug=True, solo_expert=False
        )
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
    TEST_DATA = ["neuro_trigger/tests/test_data_filter.csv"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dm = BelleIIDataManager(self.TEST_DATA)

    def test_index2mask_array(self):
        index_array = torch.tensor([1, 3, 8, 10])
        n = 11
        b_array = index2mask_array(index_array, n)
        self.assertTrue(
            (
                b_array
                == torch.tensor(
                    [
                        False,
                        True,
                        False,
                        True,
                        False,
                        False,
                        False,
                        False,
                        True,
                        False,
                        True,
                    ]
                )
            ).all()
        )

    def test_identityfilter(self):
        d = self.dm.dataset(filter=IdentityFilter())
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
        d = self.dm.dataset(
            filter=ConCatFilter(
                filters=[DuplicateEventsFilter(), Max2EventsFilter(), IdentityFilter()]
            )
        )
        le = 15
        self.assertEqual(le, len(d))


class WeightedSamplerTest(unittest.TestCase):
    # TEST_DATA = "neuro_trigger/tests/test_data_filter.csv"
    TEST_DATA = ["neuro_trigger/tests/test_data.csv"]
    PLOT = False

    def test_distuniform(self):
        dist = uniform(loc=-1, scale=2)
        n_buckets = 11
        dm = BelleIIDataManager(self.TEST_DATA)
        d = dm.dataset(
            dataset_class=partial(BelleIIDistDataset, dist=dist, n_buckets=n_buckets)
        )
        d_unchanged = dm.dataset()
        z = [i[1][0].item() for i in d]
        z_unchanged = [i[1][0].item() for i in d_unchanged]
        hist, bin_edges = np.histogram(z, n_buckets, range=(-1, 1))
        hist = hist / np.sum(hist)

        # compare to expected values
        for idx, h in enumerate(hist):
            # p = dist.cdf(bin_edges[idx+1]) - dist.cdf(bin_edges[idx])
            # p = d.get_bounds(idx)
            p = dist.cdf(d.get_bounds(idx)[1]) - dist.cdf(d.get_bounds(idx)[0])
            self.assertTrue(abs(p - h) < 0.1)
            # TODO: check if the they are actually from the same bucket

        self.assertEqual(len(z), len(d))

        if self.PLOT:
            # plot histogram
            import matplotlib.pyplot as plt

            plt.clf()
            plt.hist(z, n_buckets, range=(-1, 1), label="sampled z histogram")
            # xline = (-1, 1)
            # yline = (len(d)/n_buckets, len(d)/n_buckets)
            # plt.plot(xline, yline, color="green")
            plt.hist(z_unchanged, n_buckets, range=(-1, 1), label="real z histogram")
            xline = [
                (
                    d.get_bounds(i, inf_bounds=False)[0]
                    + d.get_bounds(i, inf_bounds=False)[1]
                )
                / 2
                for i in range(len(hist))
            ]
            yline = [i * len(d) for i in d.probs]
            # yline = (1/n_buckets, 1/n_buckets)
            plt.plot(xline, yline, color="red", label="Distribution")
            plt.xlabel("z (m)")
            plt.ylabel("count")
            plt.legend()

            plt.savefig("docs/uniform_label.png")

    def test_distnorm_inf_bounds(self):
        dist = norm(loc=0, scale=0.6)
        n_buckets = 11
        # np.random.seed(1234)
        dm = BelleIIDataManager(self.TEST_DATA)
        d = dm.dataset(
            dataset_class=partial(
                BelleIIDistDataset, dist=dist, n_buckets=n_buckets, inf_bounds=True
            )
        )
        d_unchanged = dm.dataset()
        z = [i[1][0].item() for i in d]
        z_unchanged = [i[1][0].item() for i in d_unchanged]
        hist, bin_edges = np.histogram(z, n_buckets, range=(-1, 1))
        hist = hist / np.sum(hist)
        self.assertEqual(len(z), len(d))

        # compare to expected values
        for idx, h in enumerate(hist):
            # p = dist.cdf(bin_edges[idx+1]) - dist.cdf(bin_edges[idx])
            p = dist.cdf(d.get_bounds(idx)[1]) - dist.cdf(d.get_bounds(idx)[0])
            self.assertTrue(abs(p - h) < 0.1)

        if self.PLOT:
            # plot histogram
            import matplotlib.pyplot as plt

            plt.clf()
            plt.hist(z, n_buckets, range=(-1, 1), label="sampled z histogram")
            plt.hist(z_unchanged, n_buckets, range=(-1, 1), label="real z histogram")
            xline = [
                (
                    d.get_bounds(i, inf_bounds=False)[0]
                    + d.get_bounds(i, inf_bounds=False)[1]
                )
                / 2
                for i in range(len(hist))
            ]
            yline = [i * len(d) for i in d.probs]
            plt.plot(xline, yline, color="red", label="Distribution")
            plt.xlabel("z (m)")
            plt.ylabel("count")
            plt.legend()

            plt.savefig("docs/norm_inf_bounds.png")

    def test_distnorm_non_inf_bounds(self):
        dist = norm(loc=0, scale=0.6)
        n_buckets = 11
        # np.random.seed(1234)
        dm = BelleIIDataManager(self.TEST_DATA)
        d = dm.dataset(
            dataset_class=partial(
                BelleIIDistDataset, dist=dist, n_buckets=n_buckets, inf_bounds=False
            )
        )
        d_unchanged = dm.dataset()
        z = [i[1][0].item() for i in d]
        z_unchanged = [i[1][0].item() for i in d_unchanged]
        hist, bin_edges = np.histogram(z, n_buckets, range=(-1, 1))
        hist = hist / np.sum(hist)
        self.assertEqual(len(z), len(d))

        # compare to expected values
        for idx, h in enumerate(hist):
            # p = dist.cdf(bin_edges[idx+1]) - dist.cdf(bin_edges[idx])
            p = dist.cdf(d.get_bounds(idx)[1]) - dist.cdf(d.get_bounds(idx)[0])
            self.assertTrue(abs(p - h) < 0.1)

        if self.PLOT:
            # plot histogram
            import matplotlib.pyplot as plt

            plt.clf()
            plt.hist(z, n_buckets, range=(-1, 1), label="sampled z histogram")
            plt.hist(z_unchanged, n_buckets, range=(-1, 1), label="real z histogram")
            xline = [
                (
                    d.get_bounds(i, inf_bounds=False)[0]
                    + d.get_bounds(i, inf_bounds=False)[1]
                )
                / 2
                for i in range(len(hist))
            ]
            yline = [i * len(d) for i in d.probs]
            plt.plot(xline, yline, color="red", label="Distribution")
            plt.xlabel("z (m)")
            plt.ylabel("count")
            plt.legend()

            plt.savefig("docs/norm_non_inf_bounds.png")


if __name__ == "__main__":
    unittest.main()
