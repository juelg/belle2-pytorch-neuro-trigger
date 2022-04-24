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

        # fit(trainer_module=trainers_modules[0], logger=logger)

        trainers_modules[0][0].fit(trainers_modules[0][1])

        trainers_modules = [create_trainer_pl_module(expert_i, experts, log_folder, hparams, data, version, overfit_batches=1) for expert_i in range(len(experts))]
        trainers_modules[0][0].fit(trainers_modules[0][1])

        # create dataset with predictions

        # expert_modules = [i[1] for i in trainers_modules]
        # create_dataset_with_predictions(expert_modules, path=log_folder, mode="test")
        # expert_weights_json(expert_modules, path=log_folder)

        # save_predictions_pickle(expert_modules, path=log_folder, mode="train")
        # save_predictions_pickle(expert_modules, path=log_folder, mode="val")
        # save_predictions_pickle(expert_modules, path=log_folder, mode="test")

    def test_end2end_overfit(self):
        used_config = "base"
        data = (test_data, test_data, test_data)

        hparams, log_folder, experts, version, experts_str, logger = prepare_vars(used_config, debug=True)

        trainers_modules = [create_trainer_pl_module(expert_i, experts, log_folder, hparams, data, version, overfit_batches=1) for expert_i in range(len(experts))]
        trainers_modules[0][0].fit(trainers_modules[0][1])



if __name__ == '__main__':
    unittest.main()
