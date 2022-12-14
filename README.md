# Machine Perception Project 2: Structure from Motion by John Han and Ashley Tsang

This is an implementation of Tomasi and Kanade's paper, "Shape and Motion from Image Streams under Orthography: a Factorization Method." 

NOTE FOR JOHN/ASHLEY: prior to pushing, please run 

```
rm requirements.txt
conda list -e > requirements.txt
```

to update any conda packages that were installed/uninstalled in the environment.

## Setting up Dependencies and Running Code
We used Python 3.6 and conda environment for this project. Please create a conda environment with 

```
conda create --name (env name) --file requirements.txt python=3.6
```

to install the dependencies. To run our project, type

```
python main.py --data castle
```

to run Structure from Motion using the castle dataset or

```
python main.py --data medusa
```

to run SfM using the medusa video. 