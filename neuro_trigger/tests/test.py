import unittest
import sys
sys.path.append("/mnt/scratch/juelg/neuro-trigger-v2")

from neuro_trigger.main import create_trainer_pl_module, fit, prepare_vars


# TODO
# need a subset of training data in this folder
# run fast dev run and overfit batches


test_data = "neuro_trigger/tests/test_data.csv"


class End2End(unittest.TestCase):

    def test_end2end_fast_dev_run(self):
        used_config = "base"
        data = (test_data, test_data, test_data)

        hparams, log_folder, experts, version, experts_str, logger = prepare_vars(used_config, debug=True)

        trainers_modules = [create_trainer_pl_module(expert_i, experts, log_folder, hparams, data, version, fast_dev_run=True) for expert_i in range(len(experts))]


        trainers_modules[0][0].fit(trainers_modules[0][1])


    def test_end2end_overfit(self):
        used_config = "base"
        data = (test_data, test_data, test_data)

        hparams, log_folder, experts, version, experts_str, logger = prepare_vars(used_config, debug=True)

        trainers_modules = [create_trainer_pl_module(expert_i, experts, log_folder, hparams, data, version, overfit_batches=1) for expert_i in range(len(experts))]
        trainers_modules[0][0].fit(trainers_modules[0][1])

    # TODO: tests for filters
    # complete blackbox training with 2 epochs and check if all files exist



if __name__ == '__main__':
    unittest.main()
