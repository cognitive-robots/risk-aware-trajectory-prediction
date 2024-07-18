#########################################################################################################
# Instructions for running script:                                                                      #
#                                                                                                       #    
# python visualize_examples.py --model ./models/int_ee_me/                                              #
# --checkpoint=12 --data ../data/nuScenes_test_full.pkl --node_type PEDESTRIAN --prediction_horizon 6   #
#                                                                                                       #    
# Replace with correct model, desired data, and node type                                               #            
#                                                                                                       #
#########################################################################################################
import sys
import os
import dill
import json
import argparse
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import matplotlib.patheffects as pe
from scipy import linalg
import time

sys.path.append("./Trajectron-plus-plus/trajectron")
from tqdm import tqdm
from model.model_registrar import ModelRegistrar
from model.trajectron import Trajectron
import evaluation
import utils
from utils import prediction_output_to_trajectories
from scipy.interpolate import RectBivariateSpline

seed = 0
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

parser = argparse.ArgumentParser()
parser.add_argument("--model", help="model full path", type=str)
parser.add_argument("--checkpoint", help="model checkpoint to evaluate", type=int)
parser.add_argument("--data", help="full path to data file", type=str)
parser.add_argument("--node_type", help="node type to evaluate", type=str)
parser.add_argument("--prediction_horizon", nargs='+', help="prediction horizon", type=int, default=None)
args = parser.parse_args()

def plot_trajectories(ax,
                      prediction_dict,
                      histories_dict,
                      futures_dict,
                      node_type=None,
                      line_alpha=0.7,
                      line_width=0.2,
                      edge_width=2,
                      circle_edge_width=0.5,
                      node_circle_size=1,
                      batch_num=0,
                      kde=False):

    cmap = ['k', 'r', 'y', 'g', 'b']
    # lims are [x_min, y_min, x_max, y_max]
    lims = [np.inf, np.inf, -np.inf, -np.inf]

    for node in histories_dict:
        history = histories_dict[node]
        future = futures_dict[node]
        predictions = prediction_dict[node]

        if np.isnan(history[-1]).any():
            continue

        ax.plot(history[:, 0], history[:, 1], 'k--')
        lims = [min(history[0, 0], lims[0]), min(history[0, 1], lims[1]), max(history[0, 0], lims[2]), max(history[0, 1], lims[3])]

        for sample_num in range(prediction_dict[node].shape[1]):

            if kde and predictions.shape[1] >= 50:
                line_alpha = 1
                for t in range(predictions.shape[2]):
                    sns.kdeplot(predictions[batch_num, :, t, 0], predictions[batch_num, :, t, 1],
                                ax=ax, shade=True, shade_lowest=False,
                                color=np.random.choice(cmap), alpha=0.8)

            ax.plot(predictions[batch_num, sample_num, :, 0], predictions[batch_num, sample_num, :, 1],
                    color='b',
                    # linewidth=line_width, 
                    alpha=line_alpha)

            ax.plot(future[:, 0],
                    future[:, 1],
                    'w--',
                    path_effects=[pe.Stroke(linewidth=edge_width, foreground='k'), pe.Normal()])
            lims = [min(future[-1, 0], lims[0]), min(future[-1, 1], lims[1]), max(future[-1, 0], lims[2]), max(future[-1, 1], lims[3])]

            # Current Node Position
            if (node_type and node_type == node):
                pos_color = 'c'
                print('c', node)
            else:
                pos_color = cmap[node.type.value]
            circle = plt.Circle((history[-1, 0],
                                 history[-1, 1]),
                                node_circle_size,
                                facecolor=pos_color,
                                edgecolor=pos_color,
                                lw=circle_edge_width,
                                zorder=3)
            ax.add_artist(circle)

    ax.axis('equal')
    return lims

def visualize_preds_reimp(ax,
                         prediction_output_dict,
                         dt,
                         max_hl,
                         ph,
                         t=None,
                         node_type=None,
                         robot_node=None,
                         map=None,
                         **kwargs):

    prediction_dicts, histories_dicts, futures_dicts = prediction_output_to_trajectories(prediction_output_dict,
                                                                                      dt,
                                                                                      max_hl,
                                                                                      ph,
                                                                                      map=map)

    if len(prediction_dicts.keys()) == 0:
        return
    ts_key = t if t else list(prediction_dicts.keys())[0]

    prediction_dict = prediction_dicts[ts_key]
    histories_dict = histories_dicts[ts_key]
    futures_dict = futures_dicts[ts_key]

    if node_type not in histories_dict:
        import pdb; pdb.set_trace()

    if map is not None:
        ax.imshow(map.as_image(), origin='lower', alpha=0.5)
    lims = plot_trajectories(ax, prediction_dict, histories_dict, futures_dict, node_type=node_type, *kwargs)
    plt.xlim(lims[0]-10, lims[2]+10)
    plt.ylim(lims[1]-10, lims[3]+10)


def load_model(model_dir, env, ts=100):
    model_registrar = ModelRegistrar(model_dir, 'cpu')
    model_registrar.load_models(ts)
    with open(os.path.join(model_dir, 'config.json'), 'r') as config_json:
        hyperparams = json.load(config_json)

    trajectron = Trajectron(model_registrar, hyperparams, None, 'cpu')

    trajectron.set_environment(env)
    trajectron.set_annealing_params()
    return trajectron, hyperparams


if __name__ == "__main__":
    with open(args.data, 'rb') as f:
        env = dill.load(f, encoding='latin1')

    eval_stg, hyperparams = load_model(args.model, env, ts=args.checkpoint)

    if 'override_attention_radius' in hyperparams:
        for attention_radius_override in hyperparams['override_attention_radius']:
            node_type1, node_type2, attention_radius = attention_radius_override.split(' ')
            env.attention_radius[(node_type1, node_type2)] = float(attention_radius)

    scenes = env.scenes

    print("-- Preparing Node Graph")
    for scene in tqdm(scenes):
        scene.calculate_scene_graph(env.attention_radius,
                                    hyperparams['edge_addition_filter'],
                                    hyperparams['edge_removal_filter'])

    for ph in args.prediction_horizon:
        print(f"Prediction Horizon: {ph}")
        max_hl = hyperparams['maximum_history_length']

        with torch.no_grad():
            ############### MOST LIKELY Z ###############
            eval_ade_batch_errors = np.array([])
            eval_fde_batch_errors = np.array([])
            eval_kde_batch_errors = np.array([])
            predictions_t = np.array([])
            predictions_node = np.array([])
            predictions_pl = np.array([])
            predictions_scene = np.array([])
            scene_to_predictions_dict = {}
            print("-- Evaluating Full")
            # print("-- Evaluating GMM Z Mode (Most Likely)")
            # REMOVE_LATER = 0
            for scene in tqdm(scenes):
                # if REMOVE_LATER > 2:
                #     break;
                # REMOVE_LATER += 1
                timesteps = np.arange(scene.timesteps)
                # for KDE
                predictions_full = eval_stg.predict(scene,
                                               timesteps,
                                               ph,
                                               num_samples=2000,
                                               min_future_timesteps=8,
                                               z_mode=False,
                                               gmm_mode=False,
                                               full_dist=False)
                batch_error_dict = evaluation.compute_batch_statistics(predictions_full,
                                                                       scene.dt,
                                                                       max_hl=max_hl,
                                                                       ph=ph,
                                                                       node_type_enum=env.NodeType,
                                                                       map=None,
                                                                       prune_ph_to_future=False)
                eval_kde_batch_errors = np.hstack((eval_kde_batch_errors, batch_error_dict[args.node_type]['kde']))

                # for ADE/FDE and most-likely viz
                predictions = eval_stg.predict(scene,
                                               timesteps,
                                               ph,
                                               num_samples=1,
                                               min_future_timesteps=8,
                                               z_mode=True,
                                               gmm_mode=True,
                                               full_dist=False)  # This will trigger grid sampling
                # batch_error_dict = evaluation.compute_batch_statistics(predictions,
                #                                                        scene.dt,
                #                                                        max_hl=max_hl,
                #                                                        ph=ph,
                #                                                        node_type_enum=env.NodeType,
                #                                                        map=None,
                #                                                        prune_ph_to_future=False,
                #                                                        kde=False)
                # eval_ade_batch_errors = np.hstack((eval_ade_batch_errors, batch_error_dict[args.node_type]['ade']))
                # eval_fde_batch_errors = np.hstack((eval_fde_batch_errors, batch_error_dict[args.node_type]['fde']))

                # for BOTH
                predictions_t = np.hstack((predictions_t, batch_error_dict[args.node_type]['t']))
                predictions_node = np.hstack((predictions_node, batch_error_dict[args.node_type]['node']))
                predictions_pl = np.hstack((predictions_pl, batch_error_dict[args.node_type]['path_length']))
                n_scene_batch = len(batch_error_dict[args.node_type]['path_length'])
                predictions_scene = np.hstack((predictions_scene, np.repeat(scene, n_scene_batch)))
                scene_to_predictions_dict[scene] = predictions

            k = 1000
            errors = eval_kde_batch_errors
            #remove examples that are TOO easy (stationary)
            errors_for_min = errors.copy()
            errors_for_min[predictions_pl < 1] = np.inf
            best_index = np.argmin(errors_for_min)
            worst_indices = np.argpartition(errors, -k)[-k:]

            label = "best"
            i = k
            fig, ax = plt.subplots()
            fig.set_size_inches(18.5, 10.5)
            plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
            plt.margins(0,0)
            for index in [best_index.tolist()] + worst_indices.tolist():
                t = predictions_t[index]
                node_type = predictions_node[index]
                scene = predictions_scene[index]
                predictions = scene_to_predictions_dict[scene]
                print(node_type, id(scene.map['VISUALIZATION']))
                visualize_preds_reimp(ax, predictions, scene.dt, t=t, node_type=node_type,
                                        max_hl=max_hl, ph=ph, map=scene.map['VISUALIZATION'])
                plt.savefig("{}_kde={}_{}.png".format(label, errors[index], time.time()))
                label = "worst{}".format(i)
                i -= 1
                # plt.show()
                plt.cla()
