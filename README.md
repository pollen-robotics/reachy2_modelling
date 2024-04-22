# Reachy2 robot moodelling and performance evaluation

## Installing the necessary packages manually

<details>
    <summary>Install with pip</summary><br>
        <p>Install <code>jupyter</code> to run this notebook</p>
        <pre>
            <code>pip install jupyter</code>
        </pre>
        <p>Install the <code>pinocchio</code> library (conda suggested but this works as well)</p>
        <pre>
            <code>pip install pin</code>
        </pre>
        <p>Install an additional library with robot data <code>example_robot_data</code> provided by pinocchio community as well <a href="https://github.com/Gepetto/example-robot-data">more info</a></p>
        <pre>
            <code>pip install example-robot-data</code>
        </pre>
        <p>Finally install the visualisation library <code>meshcat</code> that is compatible with pinocchio simple and powerful visualisaiton library <a href="https://pypi.org/project/meshcat/">more info</a></p>
        <pre>
            <code>pip install meshcat</code>
        </pre>
</details>

<details>
    <summary>Install with anaconda</summary><br>
        <p>Install <code>jupyter</code> to run this notebook</p>
        <pre>
            <code>conda install -c conda-forge jupyter</code>
        </pre>
        <p>Install the <code>pinocchio</code> library (conda suggested but this works as well)</p>
        <pre>
            <code>conda install -c conda-forge pinocchio</code>
        </pre>
        <p>Install an additional library with robot data <code>example_robot_data</code> provided by pinocchio community as well <a href="https://github.com/Gepetto/example-robot-data">more info</a></p>
        <pre>
            <code>conda install -c conda-forge example-robot-data</code>
        </pre>
        <p>Finally install the visualisation library <code>meshcat</code> that is compatible with pinocchio simple and powerful visualisaiton library <a href="https://pypi.org/project/meshcat/">more info</a></p>
        <pre>
            <code>pip install meshcat</code>
        </pre>
</details>


<br>


Finally install `pycapacity` for the workspace analysis
```bash
pip install pycapacity
```

For interactive visualisation
```
pip install ipywidgets
pip install ipympl
```

### Quick-start with Anaconda
The simplest way to install all the dependencies is to do it with conda from the provided `yaml` file
```
conda env create -f env.yaml
```
Then just activate it and run the jupyter
```
conda activate reachy_capacity
jupyter lab
```

## Running the code

You can then launch the juptyer notebook and run the code
```bash
jupyter lab
```
