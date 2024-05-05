# Reachy2 robot moodelling and performance evaluation

### Quick-start
You can also use [micromamba](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html) instead of `conda`.

```
# create the virtual environment
conda env create -f env.yaml

# activate it
conda activate reachy2_modelling

# install library
pip install -e .

# test it
python -c 'import reachy2_modelling as r2m; print(r2m.__file__)'

# run it!
jupyter-lab pinocchio_reachy.ipynb

# run some of the tests scripts
python reachy_fk_combined.py
```
