import torch
from torch import nn, optim, utils
import numpy as np
import sys
import os
import time
import dill
import json
import random
import pathlib
import warnings
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
from trajectron_risk import TrajectronRisk, create_stacking_model
from preprocessing_risk import EnvironmentDatasetRisk
from arg_parser_risk import args 
sys.path.append("./Trajectron-plus-plus/")
sys.path.append("./Trajectron-plus-plus/trajectron")
import visualization
import evaluation
from model.model_registrar import ModelRegistrar
from model.model_utils import cyclical_lr
from model.dataset import collate
from tensorboardX import SummaryWriter
import wandb
wandb.login()
# torch.autograd.set_detect_anomaly(True)
args.vis_every = None # not using it atm
NUM_ENSEMBLE = [0, 1, 2]

# Define sweep config
# sweep_configuration = {
#     "method": "bayes",
#     "name": "sweep",
#     "metric": {"goal": "minimize", "name": "val_loss"},
#     "parameters": {
#         'eta': {
#             # evenly-distributed logarithms 
#             'distribution': 'log_uniform_values',
#             'min': 0.01,
#             'max': 1,
#         },
#         # 'stackboost_percentage': {
#         #     # a flat distribution between 0 and 0.1
#         #     'distribution': 'q_uniform',
#         #     'min': 0.3,
#         #     'max': 0.5,
#         #     'q': 0.1,
#         # },
#     },
# }
# # Initialize sweep by passing in config.
# # (Optional) Provide a name of the project.
# sweep_id = wandb.sweep(sweep=sweep_configuration, project="my-first-sweep")

if not torch.cuda.is_available() or args.device == 'cpu':
    args.device = torch.device('cpu')
else:
    if torch.cuda.device_count() == 1:
        # If you have CUDA_VISIBLE_DEVICES set, which you should,
        # then this will prevent leftover flag arguments from
        # messing with the device allocation.
        args.device = 'cuda:0'

    args.device = torch.device(args.device)

# TODO FIX LATER - make stacking model able to be on both devices
if 'stack' in args.ensemble_method:
    args.eval_device = args.device

if args.eval_device is None:
    args.eval_device = torch.device('cpu')

# This is needed for memory pinning using a DataLoader (otherwise memory is pinned to cuda:0 by default)
torch.cuda.set_device(args.device)

if args.seed is not None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)


def main():
    # Load hyperparameters from json
    if not os.path.exists(args.conf):
        print('Config json not found!')
    with open(args.conf, 'r', encoding='utf-8') as conf_json:
        hyperparams = json.load(conf_json)

    # Add hyperparams from arguments
    hyperparams['dynamic_edges'] = args.dynamic_edges
    hyperparams['edge_state_combine_method'] = args.edge_state_combine_method
    hyperparams['edge_influence_combine_method'] = args.edge_influence_combine_method
    hyperparams['edge_addition_filter'] = args.edge_addition_filter
    hyperparams['edge_removal_filter'] = args.edge_removal_filter
    hyperparams['batch_size'] = args.batch_size
    hyperparams['k_eval'] = args.k_eval
    hyperparams['offline_scene_graph'] = args.offline_scene_graph
    hyperparams['incl_robot_node'] = args.incl_robot_node
    hyperparams['node_freq_mult_train'] = args.node_freq_mult_train
    hyperparams['node_freq_mult_eval'] = args.node_freq_mult_eval
    hyperparams['scene_freq_mult_train'] = args.scene_freq_mult_train
    hyperparams['scene_freq_mult_eval'] = args.scene_freq_mult_eval
    hyperparams['scene_freq_mult_viz'] = args.scene_freq_mult_viz
    hyperparams['edge_encoding'] = not args.no_edge_encoding
    hyperparams['use_map_encoding'] = args.map_encoding
    hyperparams['augment'] = args.augment
    hyperparams['override_attention_radius'] = args.override_attention_radius
    #---- ADDED ----
    hyperparams['heatmap_data'] = './ten_one_normalized_df_hist_all.csv'
    hyperparams['grid_data'] = './grid_info_all.csv'
    #---------------

    print('-----------------------')
    print('| TRAINING PARAMETERS |')
    print('-----------------------')
    print('| batch_size: %d' % args.batch_size)
    print('| device: %s' % args.device)
    print('| eval_device: %s' % args.eval_device)
    print('| Offline Scene Graph Calculation: %s' % args.offline_scene_graph)
    print('| EE state_combine_method: %s' % args.edge_state_combine_method)
    print('| EIE scheme: %s' % args.edge_influence_combine_method)
    print('| dynamic_edges: %s' % args.dynamic_edges)
    print('| robot node: %s' % args.incl_robot_node)
    print('| edge_addition_filter: %s' % args.edge_addition_filter)
    print('| edge_removal_filter: %s' % args.edge_removal_filter)
    print('| MHL: %s' % hyperparams['minimum_history_length'])
    print('| PH: %s' % hyperparams['prediction_horizon'])
    # #---- ADDED ----
    print('| Heatmap Data: %s' % hyperparams['heatmap_data'])
    print('| Grid Data: %s' % hyperparams['grid_data'])
    print('| Ensemble Method: %s' % args.ensemble_method)
    print('| Ensemble Models: %s' % NUM_ENSEMBLE)
    # #---------------
    print('-----------------------')

    run = wandb.init(
        # mode="disabled", # for testing
        # Set the project where this run will be logged
        project="train-risk",
        # Track hyperparameters and run metadata
        config={
            "learning_rate": hyperparams['learning_rate'],
            "epochs": args.train_epochs,
        })
    eta = None
    percentage = None
    # # uncomment if we're sweeping
    # # note that we define values from `wandb.config`
    # # instead of defining hard values
    # eta = wandb.config.eta
    # if args.ensemble_method == 'stackboost':
    #     percentage = wandb.config.stackboost_percentage

    log_writer = None
    model_dir = None

    # #---- ADDED ----
    import pandas as pd
    if hyperparams['heatmap_data']:
        heatmap_df = pd.read_csv(hyperparams['heatmap_data'])
        heatmap_tensor = torch.tensor(heatmap_df.values)

    if hyperparams['grid_data']:
        grid_df = pd.read_csv(hyperparams['grid_data'])
        grid_tensor = torch.tensor(grid_df.values)
    #---------------
    if not args.debug:
        # Create the log and model directiory if they're not present.
        model_dir = os.path.join(args.log_dir,
                                 'models_' + time.strftime('%d_%b_%Y_%H_%M_%S', time.localtime()) + args.log_tag)
        pathlib.Path(model_dir).mkdir(parents=True, exist_ok=True)

        # Save config to model directory
        with open(os.path.join(model_dir, 'config.json'), 'w') as conf_json:
            json.dump(hyperparams, conf_json)

        log_writer = SummaryWriter(log_dir=model_dir)

    # Load training and evaluation environments and scenes
    train_scenes = []
    train_data_path = os.path.join(args.data_dir, args.train_data_dict)
    with open(train_data_path, 'rb') as f:
        train_env = dill.load(f, encoding='latin1')

    for attention_radius_override in args.override_attention_radius:
        node_type1, node_type2, attention_radius = attention_radius_override.split(' ')
        train_env.attention_radius[(node_type1, node_type2)] = float(attention_radius)

    if train_env.robot_type is None and hyperparams['incl_robot_node']:
        train_env.robot_type = train_env.NodeType[0]  # TODO: Make more general, allow the user to specify?
        for scene in train_env.scenes:
            scene.add_robot_from_nodes(train_env.robot_type)

    train_scenes = train_env.scenes
    train_scenes_sample_probs = train_env.scenes_freq_mult_prop if args.scene_freq_mult_train else None

    train_dataset = EnvironmentDatasetRisk(train_env,
                                       hyperparams['state'],
                                       hyperparams['pred_state'],
                                       scene_freq_mult=hyperparams['scene_freq_mult_train'],
                                       node_freq_mult=hyperparams['node_freq_mult_train'],
                                       hyperparams=hyperparams,
                                       min_history_timesteps=hyperparams['minimum_history_length'],
                                       min_future_timesteps=hyperparams['prediction_horizon'],
                                       return_robot=not args.incl_robot_node)
    train_data_loader = dict()
    for node_type_data_set in train_dataset:
        if len(node_type_data_set) == 0:
            continue

        node_type_dataloader = utils.data.DataLoader(node_type_data_set,
                                                     collate_fn=collate,
                                                     pin_memory=False if args.device is 'cpu' else True,
                                                     batch_size=args.batch_size,
                                                     shuffle=True,
                                                     num_workers=args.preprocess_workers)
        #THIS HAS THE UNFILTERED
        train_data_loader[node_type_data_set.node_type] = node_type_dataloader

    print(f"Loaded training data from {train_data_path}")

    eval_scenes = []
    eval_scenes_sample_probs = None
    if args.eval_every is not None:
        eval_data_path = os.path.join(args.data_dir, args.eval_data_dict)
        with open(eval_data_path, 'rb') as f:
            eval_env = dill.load(f, encoding='latin1')

        for attention_radius_override in args.override_attention_radius:
            node_type1, node_type2, attention_radius = attention_radius_override.split(' ')
            eval_env.attention_radius[(node_type1, node_type2)] = float(attention_radius)

        if eval_env.robot_type is None and hyperparams['incl_robot_node']:
            eval_env.robot_type = eval_env.NodeType[0]  # TODO: Make more general, allow the user to specify?
            for scene in eval_env.scenes:
                scene.add_robot_from_nodes(eval_env.robot_type)

        eval_scenes = eval_env.scenes
        eval_scenes_sample_probs = eval_env.scenes_freq_mult_prop if args.scene_freq_mult_eval else None

        eval_dataset = EnvironmentDatasetRisk(eval_env,
                                          hyperparams['state'],
                                          hyperparams['pred_state'],
                                          scene_freq_mult=hyperparams['scene_freq_mult_eval'],
                                          node_freq_mult=hyperparams['node_freq_mult_eval'],
                                          hyperparams=hyperparams,
                                          min_history_timesteps=hyperparams['minimum_history_length'],
                                          min_future_timesteps=hyperparams['prediction_horizon'],
                                          return_robot=not args.incl_robot_node)
        eval_data_loader = dict()
        for node_type_data_set in eval_dataset:
            if len(node_type_data_set) == 0:
                continue

            node_type_dataloader = utils.data.DataLoader(node_type_data_set,
                                                         collate_fn=collate,
                                                         pin_memory=False if args.eval_device is 'cpu' else True,
                                                         batch_size=args.eval_batch_size,
                                                         shuffle=True,
                                                         num_workers=args.preprocess_workers)
            eval_data_loader[node_type_data_set.node_type] = node_type_dataloader

        print(f"Loaded evaluation data from {eval_data_path}")

    # Offline Calculate Scene Graph
    if hyperparams['offline_scene_graph'] == 'yes':
        print(f"Offline calculating scene graphs")
        for i, scene in enumerate(train_scenes):
            scene.calculate_scene_graph(train_env.attention_radius,
                                        hyperparams['edge_addition_filter'],
                                        hyperparams['edge_removal_filter'])
            print(f"Created Scene Graph for Training Scene {i}")

        for i, scene in enumerate(eval_scenes):
            scene.calculate_scene_graph(eval_env.attention_radius,
                                        hyperparams['edge_addition_filter'],
                                        hyperparams['edge_removal_filter'])
            print(f"Created Scene Graph for Evaluation Scene {i}")

    model_registrar = ModelRegistrar(model_dir, args.device)
    
    trajectron = TrajectronRisk(model_registrar,
                            hyperparams,
                            log_writer,
                            args.device)

    trajectron.set_environment(train_env, NUM_ENSEMBLE)

    # create aggregation model for stacking
    aggregation_model = None
    if 'stack' in args.ensemble_method:
        num_models = len(NUM_ENSEMBLE)
        x_size = trajectron.x_size 
        z_dim = trajectron.z_dim
        zx_dim = trajectron.zx_dim
        input_multiplier = 1 if 'cluster' in args.ensemble_method else num_models
        input_dims = zx_dim if 'cluster' in args.ensemble_method else x_size
        aggregation_model = create_stacking_model(train_env, input_dims, 
                                                  args.device, input_multiplier, num_models)

    trajectron.set_aggregation(args.ensemble_method,
            agg_models=aggregation_model, percentage=percentage, eta=eta)
    trajectron.set_annealing_params()
    print('Created Training Model.')

    eval_trajectron = None
    if args.eval_every is not None or args.vis_every is not None:
        eval_trajectron = TrajectronRisk(model_registrar,
                                     hyperparams,
                                     log_writer,
                                     args.eval_device)
        eval_trajectron.set_environment(eval_env, NUM_ENSEMBLE)
        eval_trajectron.set_aggregation(args.ensemble_method,
                agg_models=aggregation_model, percentage=percentage, eta=eta)
        eval_trajectron.set_annealing_params()
    print('Created Evaluation Model.')

    optimizer = dict()
    lr_scheduler = dict()
    for node_type in train_env.NodeType:
        if node_type not in hyperparams['pred_state']:
            continue
        optimizer[node_type] = optim.Adam([{'params': model_registrar.get_all_but_name_match('map_encoder').parameters()},
                                           {'params': model_registrar.get_name_match('map_encoder').parameters(), 'lr':0.0008}], lr=hyperparams['learning_rate'])
        # Set Learning Rate
        if hyperparams['learning_rate_style'] == 'const':
            lr_scheduler[node_type] = optim.lr_scheduler.ExponentialLR(optimizer[node_type], gamma=1.0)
        elif hyperparams['learning_rate_style'] == 'exp':
            lr_scheduler[node_type] = optim.lr_scheduler.ExponentialLR(optimizer[node_type],
                                                                       gamma=hyperparams['learning_decay_rate'])

    #################################
    #           TRAINING            #
    #################################
    curr_iter_node_type = {node_type: 0 for node_type in train_data_loader.keys()}
    for epoch in range(1, args.train_epochs + 1):
        per_trainset_loop = [None]
        if args.ensemble_method == 'gradboost':
            per_trainset_loop = NUM_ENSEMBLE
        for gradboost_index in per_trainset_loop:
            label = ''
            if gradboost_index:
                label = gradboost_index # for labeling wandb recordings

            wandb.log({"epoch{}".format(label): epoch})
            model_registrar.to(args.device)
            train_dataset.augment = args.augment
            for node_type, data_loader in train_data_loader.items():
                train_losses = []
                counts = []
                curr_iter = curr_iter_node_type[node_type]
                pbar = tqdm(data_loader, ncols=80)
                # REMOVE_LATER = 0
                for batch in pbar:
                    # if REMOVE_LATER > 3:
                    #     break;
                    # REMOVE_LATER += 1
                    trajectron.set_curr_iter(curr_iter)
                    trajectron.step_annealers(node_type)
                    optimizer[node_type].zero_grad()
                    # -------- ADDED HEATMAP_TENSOR & WANDB -------
                    train_loss = trajectron.train_loss(batch, node_type, heatmap_tensor, grid_tensor, epoch, gradboost_index)
                    train_losses.append(train_loss.item())
                    counts.append(batch[0].shape[0])
                    # -------------------------------------
                    pbar.set_description(f"Epoch {epoch}, {node_type} L: {train_loss.item():.2f}")
                    train_loss.backward()
                    # Clipping gradients.
                    if hyperparams['grad_clip'] is not None:
                        nn.utils.clip_grad_value_(model_registrar.parameters(), hyperparams['grad_clip'])
                    optimizer[node_type].step()

                    # Stepping forward the learning rate scheduler and annealers.
                    lr_scheduler[node_type].step()

                    if not args.debug:
                        log_writer.add_scalar(f"{node_type}/train/learning_rate",
                                            lr_scheduler[node_type].get_last_lr()[0],
                                            curr_iter)
                        log_writer.add_scalar(f"{node_type}/train/loss", train_loss, curr_iter)

                    curr_iter += 1
                curr_iter_node_type[node_type] = curr_iter
                wandb.log({"{} full_train_loss{}".format(node_type, label): np.average(train_losses, weights=counts)})
            train_dataset.augment = False
            if args.eval_every is not None or args.vis_every is not None:
                eval_trajectron.set_curr_iter(epoch)

            #################################
            #        VISUALIZATION          #
            #################################
            args.vis_every = None
            if args.vis_every is not None and not args.debug and epoch % args.vis_every == 0 and epoch > 0:
                max_hl = hyperparams['maximum_history_length']
                ph = hyperparams['prediction_horizon']
                with torch.no_grad():
                    # Predict random timestep to plot for train data set
                    if args.scene_freq_mult_viz:
                        scene = np.random.choice(train_scenes, p=train_scenes_sample_probs)
                    else:
                        scene = np.random.choice(train_scenes)
                    timestep = scene.sample_timesteps(1, min_future_timesteps=ph)
                    predictions = trajectron.predict(scene,
                                                    timestep,
                                                    ph,
                                                    last_model_index=gradboost_index,
                                                    min_future_timesteps=ph,
                                                    z_mode=True,
                                                    gmm_mode=True,
                                                    all_z_sep=False,
                                                    full_dist=False)

                    # Plot predicted timestep for random scene
                    fig, ax = plt.subplots(figsize=(10, 10))
                    visualization.visualize_prediction(ax,
                                                    predictions,
                                                    scene.dt,
                                                    max_hl=max_hl,
                                                    ph=ph,
                                                    map=scene.map['VISUALIZATION'] if scene.map is not None else None)
                    ax.set_title(f"{scene.name}-t: {timestep}")
                    log_writer.add_figure('train/prediction', fig, epoch)

                    model_registrar.to(args.eval_device)
                    # Predict random timestep to plot for eval data set
                    if args.scene_freq_mult_viz:
                        scene = np.random.choice(eval_scenes, p=eval_scenes_sample_probs)
                    else:
                        scene = np.random.choice(eval_scenes)
                    timestep = scene.sample_timesteps(1, min_future_timesteps=ph)
                    predictions = eval_trajectron.predict(scene,
                                                        timestep,
                                                        ph,
                                                        last_model_index=gradboost_index,
                                                        num_samples=20,
                                                        min_future_timesteps=ph,
                                                        z_mode=False,
                                                        full_dist=False)

                    # Plot predicted timestep for random scene
                    fig, ax = plt.subplots(figsize=(10, 10))
                    visualization.visualize_prediction(ax,
                                                    predictions,
                                                    scene.dt,
                                                    max_hl=max_hl,
                                                    ph=ph,
                                                    map=scene.map['VISUALIZATION'] if scene.map is not None else None)
                    ax.set_title(f"{scene.name}-t: {timestep}")
                    log_writer.add_figure('eval/prediction', fig, epoch)

                    # Predict random timestep to plot for eval data set
                    predictions = eval_trajectron.predict(scene,
                                                        timestep,
                                                        ph,
                                                        min_future_timesteps=ph,
                                                        last_model_index=gradboost_index,
                                                        z_mode=True,
                                                        gmm_mode=True,
                                                        all_z_sep=True,
                                                        full_dist=False)

                    # Plot predicted timestep for random scene
                    fig, ax = plt.subplots(figsize=(10, 10))
                    visualization.visualize_prediction(ax,
                                                    predictions,
                                                    scene.dt,
                                                    max_hl=max_hl,
                                                    ph=ph,
                                                    map=scene.map['VISUALIZATION'] if scene.map is not None else None)
                    ax.set_title(f"{scene.name}-t: {timestep}")
                    log_writer.add_figure('eval/prediction_all_z', fig, epoch)

            #################################
            #           EVALUATION          #
            #################################
            if args.eval_every is not None and not args.debug and epoch % args.eval_every == 0 and epoch > 0:
                max_hl = hyperparams['maximum_history_length']
                ph = hyperparams['prediction_horizon']
                model_registrar.to(args.eval_device)
                with torch.no_grad():
                    # Calculate evaluation loss
                    val_losses = []
                    for node_type, data_loader in eval_data_loader.items():
                        eval_losses_to_average = []
                        counts = []
                        eval_loss = []
                        print(f"Starting Evaluation @ epoch {epoch} for node type: {node_type}")
                        pbar = tqdm(data_loader, ncols=80)
                        # REMOVE_LATER = 0
                        for batch in pbar:
                            # if REMOVE_LATER > 3:
                            #     break;
                            # REMOVE_LATER += 1                        
                            eval_loss_node_type = eval_trajectron.eval_loss(batch, node_type, gradboost_index)
                            eval_losses_to_average.append(eval_loss_node_type)
                            counts.append(batch[0].shape[0])
                            pbar.set_description(f"Epoch {epoch}, {node_type} L: {eval_loss_node_type.item():.2f}")
                            eval_loss.append({node_type: {'nll': [eval_loss_node_type]}})
                            del batch

                        evaluation.log_batch_errors(eval_loss,
                                                    log_writer,
                                                    f"{node_type}/eval_loss",
                                                    epoch)
                        full_eval_loss = np.average(eval_losses_to_average, weights=counts)
                        wandb.log({"{} full_eval_loss{}".format(node_type, label): full_eval_loss})
                        val_losses.append(full_eval_loss)
                    wandb.log({"val_loss{}".format(label): np.mean(val_losses)})

                    # Predict batch timesteps for evaluation dataset evaluation
                    eval_batch_errors = []
                    # REMOVE_LATER = 0
                    for scene in tqdm(eval_scenes, desc='Sample Evaluation', ncols=80):
                        # if REMOVE_LATER > 3:
                        #     break;
                        # REMOVE_LATER += 1   
                        timesteps = scene.sample_timesteps(args.eval_batch_size)
                        mink = 20
                        predictions = eval_trajectron.predict(scene,
                                                            timesteps,
                                                            ph,
                                                            last_model_index=gradboost_index,
                                                            num_samples=mink,
                                                            min_future_timesteps=ph,
                                                            full_dist=False)

                        eval_batch_errors.append(evaluation.compute_batch_statistics(predictions,
                                                                                    scene.dt,
                                                                                    max_hl=max_hl,
                                                                                    ph=ph,
                                                                                    node_type_enum=eval_env.NodeType,
                                                                                    map=scene.map))
                    for node_type in eval_env.NodeType:
                        minFDEs = []
                        kdes = []
                        for scene_errors in eval_batch_errors:
                            fdes = np.array(scene_errors[node_type]['fde']).reshape(-1,mink)
                            minFDE = np.min(fdes, axis=1)
                            minFDEs.extend(minFDE.tolist())
                            kdes.extend(scene_errors[node_type]['kde'])
                        wandb.log({"minFDE_errors{} {}".format(label, node_type): np.mean(minFDEs)})
                        wandb.log({"KDE_errors{} {}".format(label, node_type): np.mean(kdes)})

                    # evaluation.log_batch_errors(eval_batch_errors,
                    #                             log_writer,
                    #                             'eval',
                    #                             epoch,
                    #                             bar_plot=['kde'],
                    #                             box_plot=['ade', 'fde'])

                    # Predict maximum likelihood batch timesteps for evaluation dataset evaluation
                    eval_batch_errors_ml = []
                    # REMOVE_LATER = 0
                    for scene in tqdm(eval_scenes, desc='MM Evaluation', ncols=80):
                        # if REMOVE_LATER > 3:
                        #     break;
                        # REMOVE_LATER += 1   
                        timesteps = scene.sample_timesteps(scene.timesteps)

                        predictions = eval_trajectron.predict(scene,
                                                            timesteps,
                                                            ph,
                                                            num_samples=1,
                                                            min_future_timesteps=ph,
                                                            last_model_index=gradboost_index,
                                                            z_mode=True,
                                                            gmm_mode=True,
                                                            full_dist=False)

                        eval_batch_errors_ml.append(evaluation.compute_batch_statistics(predictions,
                                                                                        scene.dt,
                                                                                        max_hl=max_hl,
                                                                                        ph=ph,
                                                                                        map=scene.map,
                                                                                        node_type_enum=eval_env.NodeType,
                                                                                        kde=False))
                    for node_type in eval_env.NodeType:
                        fdes = []
                        for scene_errors in eval_batch_errors_ml:
                            fdes.extend(scene_errors[node_type]['fde'])
                        wandb.log({"fde_ml_errors{} {}".format(label, node_type): np.mean(fdes)})

                    # evaluation.log_batch_errors(eval_batch_errors_ml,
                    #                             log_writer,
                    #                             'eval/ml',
                    #                             epoch)

            if args.save_every is not None and args.debug is False and epoch % args.save_every == 0:
                model_registrar.save_models(epoch)


if __name__ == '__main__':
    main()

# wandb.agent(sweep_id, function=main, count=4) # to sweep, uncomment this, and comment 'if name==main: main()'
