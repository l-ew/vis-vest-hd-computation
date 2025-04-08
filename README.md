# vis-vest-hd-computation

This repository contains code to reproduce the results of the article "Visual-vestibular computation of head-direction and reflexive eye movement."

## Installation

To run the code, download the repository, create a conda (https://docs.conda.io/) environment from the provided yaml file and then install the project in editable mode:

```bash
git clone https://github.com/l-ew/vis-vest-hd-computation
cd vis-vest-hd-computation
conda env create -f env.yaml
conda activate vis-vest-hd-comp
python -m pip install -e .
```

The installation was tested on a MacBook Pro (M2 chip, 16 GB memory, macOS 15.3.2). It should take only a few minutes.

## Dependencies

  - python=3.11.8
  - numpy=1.24.3
  - scipy=1.11.4
  - scikit-learn=1.3.0
  - astropy==6.1.3
  - statsmodels==0.14.0
  - wquantiles==0.6
  - pandas=1.5.3
  - matplotlib-base=3.8.0
  - seaborn=0.12.2
  - cmocean=3.0.3
  - scikit-image==0.21.0

## Demo

First, the demo data (corresponding to recordings from one wild-type mouse and one FRMD7 mouse) should be downloaded to `vis-vest-hd-computation/data.`

Executing
```bash
python src/run_embedding.py
```
will map the population activity to a two-dimensional plane using Spectral Embedding. The results are saved to `vis-vest-hd-computation/data/embedding_data` and are visualized in `vis-vest-hd-computation/figures.` For computational speed, the global scale parameter is set to a fixed value (`eps=0.03`). The run time on a MacBook Pro (M2 chip, 16 GB memory) is approximately 18 minutes.

To analyze the decoded head-direction shift and eye movements in response to visual-vestibular stimulation in bright conditions as well as the decoded head-direction shift in response to head rotations in darkness for one example wild-type mouse, run
```bash
python src/main_passive.py -d -e
```
The script saves results to `vis-vest-hd-computation/data/trial_data` and generates figures in the folder `vis-vest-hd-computation/figures.` In particular, it will create plots of the decoded head-direction shift and the eye turn similar to Fig. 1 j and l as well as Fig. 2 e and f for one example wild-type mouse in the folder `vis-vest-hd-computation/figures/TD0200/all_combinations_dark.` The run time on a MacBook Pro (M2 chip, 16 GB memory) is approximately 3 minutes.

Similarly,
```bash
python src/main_passive.py -f -d -e
```
analyzes the decoded head-direction shift in response to visual-vestibular stimulation in bright conditions and head rotations in darkness for one example FRMD7 mouse. The script will generate plots similar to Fig. 4 e and f, for the example animal, in the folder `vis-vest-hd-computation/figures/TD0194/all_combinations_dark.`

The command
 `python src/main_active.py -t 'active_grating_stim' -d`
creates a scatter plot of the decoded head-direction shift versus the actual head-direction shift during clockwise and counterclockwise scene rotations similar to Fig. 5b, for the example wild-type mouse, in the folder `vis-vest-hd-computation/figures/active_grating_stim_WT.` Further, `python src/main_active.py -t 'combined' -d` creates a scatter plot the decoded head-direction shift versus the peak head velocity in response to head-only rotations similar to Extended Data Fig. 1j, for the example wild-type mouse, in the folder `vis-vest-hd-computation/figures/combined_WT.`

## Usage

First, the data should be downloaded to `vis-vest-hd-computation/data.`

The Spectral Embedding data is already provided, but can be re-generated using 
```bash
python src/run_embedding.py
```

The following table lists commands and the figures they generate.

| Command            | Generated Figures     |
|--------------------|------------------------|
| `python examples/generate_example_figures.py`   | Fig. 1 c, d and g <br> Extended Data Fig. 1 g, i <br> Extended Data Fig. 6 b, c and e, f |
| `python src/main_passive.py`   | Fig. 1 i, j, l <br> Fig. 2 b-d and e-g, Fig. 3 b-g <br> Extended Data Fig.2 a-e, Extended Data Fig.3, Extended Data Fig. 4 a-d   |
| `python src/main_passive.py -f`   | Fig. 4 b, c and e, f <br>   |
| `python src/main_passive.py -d`  <br> `python src/main_passive.py -f -d`  <br> `python src/vis_vest_hd_computation/compare_WT_FRMD7.py` | Fig. 4 h <br> Extended Data Fig. 5 a  |
| `python src/main_active.py -t 'combined'` | Extended Data Fig. 1 f, h, j  |
| `python src/main_active.py -f -t 'active_3landmarks'` | Extended Data Fig. 6 d, g  |
| `python src/main_active.py -t 'active_grating_stim'` | Fig. 5 b, c  |

## License

Apache License 2.0