# Deep RL Quadcopter Controller

*Teach a Quadcopter How to Fly!*

In this project, I designed a reinforcement agent to fly a quadcopter. I chose deep deterministic policy gradient ([DDPG](https://arxiv.org/abs/1509.02971)) algorithm, which suits well the the continuous control problem of a quadcopter.

## Implementations

My DDPG agent is implemented using Tensorflow and Keras, in the folder `agents/`,
- `ddpg.py`: DDPG agent, experience replay buffer, Ornstein-Uhlenbeck noise.
- `actor_critic.py`: actor and critic neural network models.
- `plot_utils.py`: utility functions for plotting the results.
- `task.py`: task for the quadcopter.


## Project Instructions

1. Clone the repository and navigate to the downloaded folder.

```
git clone https://github.com/yingzwang/RL-Quadcopter-2.git
cd RL-Quadcopter-2
```

2. Create and activate a new environment.

```
conda create -n quadcop python=3.6 matplotlib numpy pandas
source activate quadcop
```

3. Create an [IPython kernel](http://ipython.readthedocs.io/en/stable/install/kernel_install.html) for the `quadcop` environment. 
```
python -m ipykernel install --user --name quadcop --display-name "quadcop"
```

4. Open the notebook.
```
jupyter notebook Quadcopter_Project.ipynb
```

5. Before running code, change the kernel to match the `quadcop` environment by using the drop-down menu (**Kernel > Change kernel > quadcop**). Then, follow the instructions in the notebook.

6. You will likely need to install more pip packages to complete this project.  Please curate the list of packages needed to run your project in the `requirements.txt` file in the repository.
