# MetaGenRL: Improving Generalization in Meta Reinforcement Learning using Learned Objectives

This is the official research code for the paper Kirsch et al. 2019:
"Improving Generalization in Meta Reinforcement Learning using Learned Objectives".

## Installation

Install the following dependencies (in a virtualenv preferably)
```bash
pip3 install ray[tune]==0.7.7 gym[all] mujoco_py>=2 tensorflow-gpu==1.13.2 scipy numpy
```

This code base uses [ray](https://github.com/ray-project/ray), if you would like to use multiple machines,
see [the ray documentation for details](https://ray.readthedocs.io/en/latest/using-ray-on-a-cluster.html).

We also make use of ray's native tensorflow ops. Please compile them by running
```bash
python3 -c 'import ray; from pyarrow import plasma as plasma; plasma.build_plasma_tensorflow_op()'
```

## Meta Training

Adapt the configuration in `ray_experiments.py` (or use the default configuration) and run

```bash
python3 ray_experiments.py train
```

By default, this requires a local machine with 4 GPUs to run 20 agents in parallel.
Alternatively, skip this and download a pre-trained objective function as described below.

## Meta Testing

After running meta-training (or downloading a pre-trained objective function)
you can train a new agent from scratch on an environment of your choice.
Optionally configure your training in `ray_experiments.py`, then run

```bash
python3 ray_experiments.py test --objective TRAINING_DIRECTORY
```

This only requires a single GPU on your machine.

## Using a pre-trained objective function

Download a pre-trained objective function,

```bash
cd ~/ray_results/metagenrl
curl https://github.com/timediv/metagenrl/releases/download/pretrained-v1/CheetahLunar.tgz|tar xvz
```

and proceed with meta testing as above.
In this case your `TRAINING_DIRECTORY` will be `pretrained-CheetahLunar`.

## Visualization

Many tf summaries are written during training and testing and can be visualized with tensorboard
```bash
tensorboard --logdir ~/ray_results/metagenrl
```
