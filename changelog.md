# Changelog

## Version 0.4
- [ ] baseline model v2 with BN and Relu
- [ ] add categories in config
- [ ] reweighting of trainings sample, by duplicating samples per bin or by reweighting them per bin -> or random sampling with same prob. per bin, idea: make classification problem
- [ ] train with different batchsizes and learning rates per expert
- [ ] reimplement dataset caching


## Version 0.3
- [ ] make an average loss graph from all experts -> how to communicate with the other experts? -> easy cause we use threads -> use class where all log to and if all have logged for an epoch we can log to tensorboard and create visualizations
- [x] add statistical values such as mean and std to the plots (in form of legends)
- [x] implement rprop, generalize optimizers and put them into config
- [x] fix weight init
- [x] act function into the config
- [x] fix x axis of hist plot for gt data
- [x] add diff hist plot -> z(Reco-Neuro)
- [x] add std(z(Reco-Neuro)) to tensorboard plot metrics and relative old vs new
- [x] add std bins plot
- [x] pin pytorch lightning version
- [x] rescale z/theta outputs to represent real physical values
- [ ] plot creation for already trained weights -> write cli where weights are loaded
- [ ] in the end of the training create weights, predication dataset, plots as pngs, and maybe evaluate test?
- [ ] only train on the first reco track of a track

## Version 0.2
- [x] description in experiment log
- [x] add extending of other config
- [x] add training for only z component
- [x] add easy dict
- [x] save git diff and commit id into log
- [x] add change log
- [x] per expert hparams -> which overwrite the default value e.g. different bach sizes
- [x] issue with singal shutdown in thread
- [x] models and critic function also in configuration using dicts