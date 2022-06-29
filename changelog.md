# Changelog
Versions are tagged in git under v0.x

# Ideas for reweighting

## Oversampling



## Version 0.4
- [ ] plot creation for already trained weights -> write cli where weights are loaded
- [ ] only train on the first reco track of a track
- [ ] make an average loss graph from all experts -> how to communicate with the other experts? -> easy cause we use threads -> use class where all log to and if all have logged for an epoch we can log to tensorboard and create visualizations
- [ ] distributions into the same plot to make them more comparable
- [ ] baseline model v2 with BN and Relu
- [ ] add gradient clipping
- [ ] add categories in config
- [ ] reweighting of trainings sample, by duplicating samples per bin or by reweighting them per bin -> or random sampling with same prob. per bin, idea: make classification problem
- [x] train with different batchsizes and learning rates per expert
- [ ] write readme page
- [x] add unit tests
- [ ] add linting
- [ ] add bitbucket pipeline
- [x] create pickle file with z, theta predictions after training for future comparision
- [x] dont use dataset predictions but optionally the ones from older trainings
- [ ] able to compare not only with one specified, but also with others -> that should probably be a command line tool (after training)
- [x] file with single output number -> over all experts and per expert and maybe compare to previous
- [x] distribution sampling with in config: distribution should be configurable
- [x] organize main better and support cmd args
- [x] global experiment log
- [x] filter functions
- [x] native filter datasets
- [ ] config as python objects


ideas from kai
- [x] include softsign activation function
- [ ] bigger networks that where pruned
- [ ] more weights to z, but not really an improvement
- [ ] classification (if displaced vertex or not)
- [ ] cnn -> only 406 parameters and good accuracy
- [ ] lstm -> but is there time data? -> 64 clock cycle


## Version 0.3
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
- [x] reimplement dataset caching
- [x] in the end of the training create weights, predication dataset, plots as pngs, and maybe evaluate test?
- [x] validate with best trained data
- [x] export (the best) weights in the end of the training

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