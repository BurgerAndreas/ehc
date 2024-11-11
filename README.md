# ehc

Run in interactive session
```bash
srun -c 4 --gres=gpu:rtx6000:1 --mem=16GB --pty --time=2-00:00:00 --qos=long bash

mamba activate ehc
module load cuda-11.8

cd LapNet
python3 main.py --config lapnet/configs/ferminet_systems.py --config.system.molecule_name CH4
python3 main.py --config lapnet/configs/benzene_dimer/benzene_dimer.py:4.95 --config.pretrain.iterations 50000 --config.pretrain.basis augccpvd --config.optim.forward_laplacian=False
```

Run via sbatch
```bash
cd ehc
sbatch launchslurm --config lapnet/configs/ferminet_systems.py --config.system.molecule_name CH4
sbatch launchslurm --config lapnet/configs/benzene_dimer/benzene_dimer.py:4.95 --config.pretrain.iterations 50000 --config.pretrain.basis augccpvd --config.optim.forward_laplacian=False
```

## Install on slurm cluster (Vector for example)

Optional but recommended: Mamba, the better conda
```bash
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh
source ~/.bashrc
~/miniforge3/bin/mamba
source ~/.bashrc
```

```bash
mkdir ehc; cd ehc

module avail

mamba deactivate
mamba remove --name ehc --all
mamba create -n ehc python=3.10 -y
mamba activate ehc
module load cuda-11.8
# we want: cuda11/jaxlib-0.3.24+cuda11.cudnn805-cp38-cp38-manylinux2014_x86_64.whl
pip3 install --upgrade jax[cuda]==0.3.24 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# numpy==1.24.4
pip install numpy==1.21.5 scipy==1.7.3 matplotlib seaborn requests tensorboard_plugin_profile

git clone https://github.com/YWolfeee/lapjax.git
pip install ./lapjax

git clone https://github.com/bytedance/LapNet.git
pip install ./LapNet

# something about
# LapNet: when wrapping `jax` modules `collect_profile.py`, got importerror: this script requires `tensorflow` to be installed.
# https://www.tensorflow.org/install/source#gpu
# we want: https://storage.googleapis.com/tensorflow/versions/2.13.0/tensorflow-2.13.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
pip install tensorflow[and-cuda]==2.12
pip install tensorrt==8.5.3.1
```
```bash
git clone git@github.com:BurgerAndreas/ehc.git
cd ehc
```
