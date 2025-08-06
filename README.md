# [Adaptive Trust Region Radius for Robust Policy Optimization](https://link.springer.com/chapter/10.1007/978-981-96-6951-6_27)

This is the official implementation of the paper "Adaptive Trust Region Radius for Robust Policy Optimization."

## File Structure

```
root/
├── atrpo                   # Core algorithm implementation
├── conf                    # Config files for Hydra
├── notebook                # Notebooks for plotting results
├── optimization_results    # Results of HPO
├── scripts                 # Utility scripts
├── .python-version         # Python version (e.g. 3.10.12)
├── hpo.py                  # Main script for HPO
├── main.py                 # Main experiment runner
├── poetry.lock             # Locked dependencies
├── pyproject.toml          # Poetry project config
├── LICENSE                 # MIT License
└── README.md               # This file


```

## Setup

-   Clone this repository
    -   `git clone https://github.com/shiralab/adaptive-trpo.git`
-   [Install pyenv](https://github.com/pyenv/pyenv-installer)
-   Install python

```bash
pyenv install 3.10.12
pyenv local 3.10.12
```

-   [Install poetry](https://python-poetry.org/docs/)
-   Install dependencies
    -   Activate the virtual environment: `poetry shell`
    -   **Note:** gym 0.21 has known installation issues.

```bash
pip install setuptools==65.5.0 pip==21  # gym 0.21 installation is broken with more recent versions
pip install wheel==0.38.0
pip install gym==0.21.0
poetry install
```

-   Reinstall CUDA-enabled builds for PyTorch and JAX
    -   Check CUDA version: `nvcc -V`
    -   Check currently installed torch version: `pip show torch`
    -   Reinstall PyTorch with CUDA support:
        -    Follow the instructions at https://pytorch.org/get-started/previous-versions/
            -    Replace `<torch_version>` and `<cu_version>` (e.g., `cu118`) with your target versions.
            -    `pip install --upgrade --no-deps --force-reinstall torch==<torch_version> --index-url https://download.pytorch.org/whl/<cu_version>`
    -   Reinstall JAX with CUDA support:
        -    Follow the instructions at https://jax.readthedocs.io/en/latest/installation.html
-   (Optional) Login to Weights & Biases (W&B):
    -   `wandb login`

## How To Reproduce

### Experiment 1

Run over multiple `algorithm.target_kl` (trust region radius) values on a given environment.
Hydra will automatically launch one run per comma-separated value.
For example, to run on the `hopper` environment while sweeping the `algorithm.target_kl` values over 10 random seeds, run:

```bash
python main.py -m \
  env=hopper \
  algorithm=trpo \
  algorithm.n_steps=1024 \
  algorithm.target_kl=1.0,0.5,0.1,0.05,0.01,0.005,0.001,0.0005,0.0001 \
  experiment.total_timesteps=2000000 \
  experiment.n_envs=2 \
  experiment.n_eval_envs=20 \
  experiment.gpu_id_algo=0 \
  experiment.gpu_id_env=0 \
  experiment.seed="range(0,10)"
```

Adjust `env` and `algorithm.target_kl` as needed.
Other parameters (e.g., `experiment.n_eval_envs`, `experiment.gpu_id_algo`) can be overridden similarly.

### Experiment 2

#### Hyperparameter Optimization

Use `hpo.py` to search the hyperparameter space defined in `conf/search_space/ax/*.yaml`.
For example, to perform hyperparameter optimization on `hopper` using ATRPO method, run:

```bash
python hpo.py -m \
  env=hopper \
  algorithm=adaptive_trpo \
  algorithm.n_steps=1024 \
  experiment.total_timesteps=2000000 \
  experiment.n_envs=2 \
  experiment.n_eval_envs=20
```

Hydra outputs the best configuration as the YAML file in its output directory.
Check the path and move or copy the file as needed.

#### Testing The Best Hyperparameters

Plug the discovered hyperparameters into `main.py` and rerun as in Experiment 1.
For example, to test the discovered ATRPO hyperparameters on the `hopper` environment over 10 random seeds, run:

```bash
python main.py -m \
  env=hopper \
  algorithm=adaptive_trpo \
  algorithm.n_steps=1024 \
  algorithm.target_kl=0.0012834259559537453 \
  algorithm.smoothing_coeff_func.const=0.12874209135770798 \
  algorithm.kl_update_ratio_func.target_snr=2.11995891854167 \
  algorithm.kl_update_ratio_func.kl_update_strength_func.const=0.06840647444780916 \
  experiment.total_timesteps=2000000 \
  experiment.n_envs=2 \
  experiment.n_eval_envs=20 \
  experiment.gpu_id_algo=0 \
  experiment.gpu_id_env=0 \
  experiment.seed="range(4,14)"
```

### Visualization

1. Use `notebook/convert_to_csv.ipynb` to convert W&B or MLflow logs into csv files.
1. (Optional) Execute `scripts/random_policy.py` to collect performance of random policies (used for calculating performance metrics)
1. Use `notebook/result_plotter.ipynb` to generate figures from CSV files.
1. Open `notebook/fanova.ipynb` for fANOVA visualization of HPO results.
    - Note: `fanova` dependencies are not installed by Poetry; create a seperate virtual environment if needed.

## Citation

```
@InProceedings{shimizu2025atrpo,
  author={Shimizu, Shoma and Uchida, Kento and Maki, Atsuo and Shirakawa, Shinichi},
  editor={Mahmud, Mufti and Doborjeh, Maryam and Wong, Kevin and Leung, Andrew Chi Sing and Doborjeh, Zohreh and Tanveer, M.},
  title={{Adaptive Trust Region Radius for Robust Policy Optimization}},
  booktitle={Neural Information Processing},
  year={2025},
  publisher={Springer Nature Singapore},
  address={Singapore},
  pages={387--404},
  isbn={978-981-96-6951-6}
}
```
