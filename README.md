# Risk-aware Trajectory Prediction by Incorporating Spatio-temporal Traffic Interaction Analysis

This repository corresponds to the official source code of the Risk-aware Trajectory Prediction by Incorporating Spatio-temporal Traffic Interaction Analysis paper:

<a href="TODO arxiv link">Risk-aware Trajectory Prediction by Incorporating Spatio-temporal Traffic Interaction Analysis</a>

### Download
1. Clone the git repository
2. Checkout the ```eccv2020``` branch
3. From the ```4yp-roadrisk/``` directory , download the <a href="https://github.com/StanfordASL/Trajectron-plus-plus">Trajectron++ code</a> with

    ```git clone --recurse-submodules git@github.com:StanfordASL/Trajectron-plus-plus.git```
5. Create the trajectron++ conda environment (instructions are within <a href="https://github.com/StanfordASL/Trajectron-plus-plus">Trajectron++ repo</a>)
6. Make folders for the data and models by running

    ```mkdir data models```

   from the ```4yp-roadrisk/``` directory 

### Data

#### ETH-UCY: 

#### nuScenes:
1. Follow the instructions within the Trajectron++ repository to download the full nuScenes dataset into the ```4yp-roadrisk/Trajectron_plus_plus/experiments/nuScenes``` folder
2. activate the trajectron++ conda environment
3. from the ```4yp-roadrisk/``` directory run

    ```python process_data.py --data=./Trajectron-plus-plus/experiments/nuScenes/v1.0 --version="v1.0-trainval" --output_path=./data```

### Training
Run


```python train_risk.py --eval_every 1 --vis_every 1 --conf ./Trajectron-plus-plus/experiments/nuScenes/models/int_ee_me/config.json --data_dir ./data/ --train_data_dict nuScenes_train_full_eccv2020_risk.pkl --eval_data_dict nuScenes_val_full_eccv2020_risk.pkl --offline_scene_graph yes --preprocess_workers 10 --batch_size 256 --log_dir ./models --train_epochs 20 --node_freq_mult_train --log_tag testing_4yp_eccv2020_riskdata_int_ee_me --map_encoding --augment```



### Pre-trained Models
Pretrained models are provided under ```models/```. 

### Testing
Run 
