# REMAKE Experiment Pipeline

Minimal research pipeline:

1. Run `notebooks/predict.ipynb` to train price/load/PV LSTM artifacts and build `share_data`.
2. Run `notebooks/madrl.ipynb` to train and evaluate `MADRL_BASE`, `MADRL_PENALTY`, `MADRL_PROJECTION`.
3. Run `notebooks/misocp.ipynb` and `notebooks/mpc.ipynb` to save cached MISOCP, LOCAL_MPC, and ADMM_MPC results.
4. Run `notebooks/compare.ipynb` to read cached results and write tables/figures.

The code reads copied CSV files from `REMAKE/datasets/prosumer`, defaults deep-learning training to `cuda`, and keeps ordinary missing-key/file/column failures natural. Long-running LSTM, MADRL, MPC, MISOCP, and ADMM loops use ASCII `tqdm` progress bars.
