# Risk-aware Trajectory Prediction by Incorporating Spatio-temporal Traffic Interaction Analysis

This repository corresponds to the official source code of the paper:

<a href="TODO arxiv link">Risk-aware Trajectory Prediction by Incorporating Spatio-temporal Traffic Interaction Analysis</a>

### Repo Depenencies
1. Clone the git repository
2. Checkout the ```eccv2020``` branch
3. From the ```4yp-roadrisk/``` directory , download the <a href="https://github.com/StanfordASL/Trajectron-plus-plus">Trajectron++ code</a> with

    ```git clone --recurse-submodules git@github.com:StanfordASL/Trajectron-plus-plus.git```
5. Create the trajectron++ conda environment (instructions are within <a href="https://github.com/StanfordASL/Trajectron-plus-plus">Trajectron++ repo</a>)
6. Make folders for the data by running

    ```mkdir data```

   from the ```4yp-roadrisk/``` directory 

### Data

1. Follow the instructions within the Trajectron++ repository to download the full nuScenes dataset into the ```4yp-roadrisk/Trajectron_plus_plus/experiments/nuScenes``` folder (to do this within commandline, follow https://github.com/nutonomy/nuscenes-devkit/issues/110)
3. activate the trajectron++ conda environment
4. from the ```4yp-roadrisk/``` directory run

    ```python process_data.py --data=./Trajectron-plus-plus/experiments/nuScenes/v1.0 --version="v1.0-trainval" --output_path=./data```

### Training
Run
```python train_risk.py --eval_every 1 --vis_every 1 --conf ./Trajectron-plus-plus/experiments/nuScenes/models/int_ee_me/config.json --data_dir ./data/ --train_data_dict nuScenes_train_full.pkl --eval_data_dict nuScenes_val_full.pkl --offline_scene_graph yes --preprocess_workers 10 --batch_size 256 --log_dir ./models --train_epochs 20 --node_freq_mult_train --log_tag testing_int_ee_me --map_encoding --augment```

with flag `--location_risk` for Location-Risk model, `--no_stationary` for No-Stationary model, or both for Location-Risk+No-Stationary model



### Pre-trained Models
Pretrained models are provided under ```models/```. 

### Testing
Run ```python evaluate_nuScenes_risk.py --model models/location_risk_int_ee_me/ --checkpoint=12 --data ./data/nuScenes_test_full.pkl --output_path ./results/ --output_tag location_risk_int_ee_me --node_type VEHICLE --prediction_horizon 2 4 6 8```

for any of the models within `models/` (tip - make sure to change the output_tag as well).
