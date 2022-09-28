# Simulation-based inference for exoplanetary atmospheric retrieval

## Installation

1. Clone the repository.

    ```
    git clone https://github.com/francois-rozet/sbi-ear
    cd sbi-ear
    ```

2. Download and extract the input data of [petitRADTRANS](https://petitradtrans.readthedocs.io).

    ```
    wget https://keeper.mpdl.mpg.de/f/78b3c66857924b5aacdd/?dl=1 -O input_data.tar.gz
    tar -xzf input_data.tar.gz
    ```

3. Create and activate the `conda` environment.

    ```
    conda env create -f environment.yml
    conda activate ear
    ```

4. Rebin the opacities to a lower resolution.

    ```
    python rebin.py
    ```

## Experiments

Running the experiment scripts requires a [Slurm](https://wikipedia.org/wiki/Slurm_Workload_Manager) cluster.

1. Generate the training, validation and testing data.

    ```
    python generate.py
    ```

2. When the data generation is finished, launch the training of the estimator.

    ```
    python train.py
    ```

    > This step requires to login to [Weights & Biases](https://wandb.ai/site).

3. Run the evaluation notebook.

    ```
    jupyter notebook eval.ipynb
    ```

    > It is necessary to modify the `runpath` according to the run name.
