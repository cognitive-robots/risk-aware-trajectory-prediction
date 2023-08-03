# TODO Paper Name

This repository corresponds to the official source code of the TODO conference_name paper:

<a href="TODO arxiv link">TODO Paper Name</a>

### Download
1. Clone the git repository
2. Checkout the ```eccv2020``` branch
3. From the ```4yp-roadrisk/``` directory , download the <a href="https://github.com/StanfordASL/Trajectron-plus-plus">Trajectron++ code</a> with `git clone --recurse-submodules git@github.com:StanfordASL/Trajectron-plus-plus.git`
4. Create the trajectron++ conda environment (instructions are within <a href="https://github.com/StanfordASL/Trajectron-plus-plus">Trajectron++ repo</a>)
5. Follow the instructions within the Trajectron++ repository to download the full nuScenes dataset into the ```4yp-roadrisk/Trajectron_plus_plus/experiments/nuScenes``` folder
6. Make folders for the data and models by running ```mkdir data models``` from the ```4yp-roadrisk/``` directory 

### Data

#### ETH-UCY: 

#### nuScenes:
1. activate the trajectron++ conda environment
2. from the ```4yp-roadrisk/``` directory run ```python process_data.py --data=./Trajectron-plus-plus/experiments/nuScenes/v1.0 --version="v1.0-trainval" --output_path=./data```

### Training

### Pre-trained Models
Pretrained models are provided under ```models/```. 

### Testing
Run 

