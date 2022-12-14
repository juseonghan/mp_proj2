# Machine Perception Project 2: Structure from Motion by John Han and Ashley Tsang

This is an implementation of Tomasi and Kanade's paper, "Shape and Motion from Image Streams under Orthography: a Factorization Method." 


## Setting up Dependencies and Running Code
We used Python 3.6 and conda environment for this project. To run our code and install the dependencies, please run

```
conda create --name env_name python=3.6
chmod +x dependencies.sh
./dependencies.sh
```

Afterwards, to run our project, go into the src directory and run

```
python main.py --data castle --points 800
```

to run Structure from Motion using the castle dataset or

```
python main.py --data medusa --points 800
```

to run SfM using the medusa video. In order to visualize the results, enter into the root directory and run `visualize.py` and specify visualization of dataset and motion or of shape. For example, 

```
python src/visualize.py --dataset medusa --type R
```

would visualize the camera motion for the medusa dataset, whereas

```
python src/visualize.py --data castle --type S
```

would visualize the shape point cloud for the castle dataset. 