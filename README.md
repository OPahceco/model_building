# Dark QCD Model Building

## Installation

```bash
git clone git@github.com:cms-svj/model_building
cd model_building
./install.sh
```

The `install.sh` script has a few options:
```
Options:
-e [external(s)]    (re)install specific external(s) (options = pythia8 python_packages delphes)
-p [version]        Pythia8.3 minor version to install (default = 10)
-s [protocol]       use protocol to clone (default = ssh, alternative = https)
-h                  print this message and exit
```

## Environment

```bash
source init.sh
```

This repository is built on the [LCG 106](https://lcginfo.cern.ch/release_packages/106/x86_64-el9-gcc13-opt/) environment.
Important software versions:
* Python 3.11.9
* Pythia8 3.10
* Delphes 3.5.2pre02 (patched)
* HepMC 3.2.7
* ROOT 6.32.02
* numpy 1.26.4
* coffea 2025.12.0
* uproot 5.7.5
* awkward 2.10.0
* vector 1.8.1
* matplotlib 3.11.0
* mplhep 1.3.3

## Predefined model configurations

A few predefined model configuration files are provided in the [configs](./configs) folder:
1. The CMS model used in [EXO-19-020](https://arxiv.org/abs/2112.11125) (largely based on [arXiv:1503.00009](https://www.arxiv.org/abs/1503.00009) and [arXiv:1707.05326](https://arxiv.org/abs/1707.05326)).
2. One of the Snowmass benchmark models from [arXiv:2203.09503](https://arxiv.org/abs/2203.09503).
3. The new flavor-changing dark current (FCDC) model, in both complete and simplified versions.

## Running

An example run command is:
```bash
./run_model helper -C configs/model_cms.py --steps all --events 10 --verbose
```

This command will, in order:
1. load the CMS model configuration
2. generate the Pythia and Delphes cards for this configuration
3. run Pythia
4. run Delphes
5. compute additional observables of the large-radius jets (N2, N3, and primary Lund-plane points), save them in a friend tree (`events_friend.root`), and merge the friend tree into the Delphes tree (`events_merged.root`); the merge can be skipped with `--no-merge`
6. make histograms (if `events_friend.root` exists, i.e. item 5 was run, this also includes histograms of the additional observables)

If the `--steps` command is omitted, items 3 through 6 will be skipped (the default is just to generate the cards).
Items 2 through 6 can be run separately by specifying just one of them in `--steps`, except that the Pythia and Delphes steps (items 3 and 4) also need the cards step (item 2) in the same command, e.g. `--steps cards delphes`.
Item 5 requires jet constituents, so it is skipped if `--no-constituents` is used.
Item 6 reads `events.root` and works without item 5; the histograms of the additional observables are only made if `events_friend.root` is found.

The input model configuration can be modified using command-line arguments, and the resulting configuration file will be generated along with the Pythia and Delphes cards.

Once histograms are created, an example of how to plot them, comparing different models, can be found in [Plots.py](./Plots.py).

### Alternative method

An external Pythia card can be used (instead of generating a model by providing the helper with a parameter configuration):
```bash
./run_model external --card my_card.cmnd --stableIDs 53 4900211 --darkHadronIDs 4900111 4900113 4900211 4900213 --darkHadronFinalIDs 4900111 4900211 --pythia '' --steps all --events 10 --verbose
```

As shown, the lists of stable particle IDs, dark hadron IDs, and final dark hadron IDs must be provided manually in order for the Delphes output to be correct.
The lists of dark quark and dark gluon IDs (via `--darkQuarkIDs` and `--darkGluonIDs`) should also be updated if relevant; the default values are `[4900101]` and `[4900021]`, respectively.
(The argument `--pythia ''` prevents appending common settings to the Pythia card, which are included by default.)

## Analysis

The notebook [example_MT.ipynb](./example_MT.ipynb) shows an example of interactively analyzing the generated events using [coffea](https://github.com/CoffeaTeam/coffea/):
* applying selections
* computing new quantities
* filling histograms
* making plots

For programmatic analysis, follow [Histogram.py](./Histogram.py) and [Plots.py](./Plots.py) as noted above.
