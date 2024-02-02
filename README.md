# Intro
In this repo an algorithm of batched mean reciprocal rank (MRR) evaluation is provided. 

This enables significant speed-up compared with the for-looped evaluation, e.g., in GraphGPS.

See the notebook for a running example. 

# Runtime

| batch size | sequential (s) | ours (s) |
| ---        |----------------|----------|
| 16 | 0.06           | 0.01     |
| 128 | 0.41           | 0.04     |
| 1024 | 3.03           | 0.23     |


# Environment:  
```
conda install pytorch==2.1.1 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install torch_geometric
pip install torch_scatter -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
pip install jupyterlab
```

# Reference:  
https://github.com/toenshoff/LRGB/blob/main/graphgps/head/inductive_edge.py

https://github.com/rampasek/GraphGPS/blob/main/graphgps/head/inductive_edge.py

https://github.com/snap-stanford/ogb/blob/master/ogb/linkproppred/evaluate.py