# Reachy2 robot moodelling and performance evaluation

### Quick-start
You can also use [micromamba](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html) instead of `conda`.

```
# create the virtual environment
# this also performs a pip install -e .,
# which installs the lib reachy2_modelling
conda env create -f env.yaml

# activate it
conda activate reachy2_modelling

# install library
# NOTE: no need for this if using conda env
pip install -e .

# test it
python -c 'import reachy2_modelling as r2m; print(r2m.__file__)'

# run it!
jupyter-lab pinocchio_reachy.ipynb

# run some of the tests scripts
python tests/test_reachy_fk_old.py
```
