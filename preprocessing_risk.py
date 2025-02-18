import torch
import numpy as np
import collections.abc
from torch.utils.data._utils.collate import default_collate
import dill
import sys
sys.path.append("./Trajectron-plus-plus/trajectron")
from model.dataset.preprocessing import collate, get_relative_robot_traj
from model.dataset.dataset import EnvironmentDataset, NodeTypeDataset
sys.path.append("../UniTraj-private/vicreg")
from get_embeddings import get_embeddings

import pickle
import sklearn
import matplotlib.pyplot as plt

d_clusterpcagm = pickle.load(open('clusterpcagm_colormaponly.pkl', 'rb'))
GM = d_clusterpcagm['gm']
PCA = d_clusterpcagm['clusterpca']

class EnvironmentDatasetRisk(EnvironmentDataset):
    def __init__(self, env, state, pred_state, node_freq_mult, scene_freq_mult, hyperparams, **kwargs):
        self.env = env
        self.state = state
        self.pred_state = pred_state
        self.hyperparams = hyperparams
        self.max_ht = self.hyperparams['maximum_history_length']
        self.max_ft = kwargs['min_future_timesteps']
        self.node_type_datasets = list()
        self._augment = False
        for node_type in env.NodeType:
            if node_type not in hyperparams['pred_state']:
                continue
            self.node_type_datasets.append(NodeTypeDatasetRisk(env, node_type, state, pred_state, node_freq_mult,
                                                           scene_freq_mult, hyperparams, **kwargs))

class NodeTypeDatasetRisk(NodeTypeDataset):
    def __getitem__(self, i):
        (scene, t, node) = self.index[i]

        if self.augment:
            scene = scene.augment()
            node = scene.get_node_by_id(node.id)

        return get_node_timestep_data(self.env, scene, t, node, self.state, self.pred_state,
                                      self.edge_types, self.max_ht, self.max_ft, self.hyperparams)

def get_node_timestep_data(env, scene, t, node, state, pred_state,
                           edge_types, max_ht, max_ft, hyperparams,
                           scene_graph=None):
    """
    Pre-processes the data for a single batch element: node state over time for a specific time in a specific scene
    as well as the neighbour data for it.

    :param env: Environment
    :param scene: Scene
    :param t: Timestep in scene
    :param node: Node
    :param state: Specification of the node state
    :param pred_state: Specification of the prediction state
    :param edge_types: List of all Edge Types for which neighbours are pre-processed
    :param max_ht: Maximum history timesteps
    :param max_ft: Maximum future timesteps (prediction horizon)
    :param hyperparams: Model hyperparameters
    :param scene_graph: If scene graph was already computed for this scene and time you can pass it here
    :return: Batch Element
    """

    # Node
    timestep_range_x = np.array([t - max_ht, t])
    timestep_range_y = np.array([t + 1, t + max_ft])
    x = node.get(timestep_range_x, state[node.type])
    y = node.get(timestep_range_y, pred_state[node.type])
    #--------------ADDED--------------
    unf_state = {'PEDESTRIAN': {'unfiltered_position': ['x','y']},
                 'VEHICLE': {'unfiltered_position': ['x','y']}
                }
    unf_x = node.get(timestep_range_x, unf_state[node.type])
    #DUBULAR!!
    #---------------------------------
    first_history_index = (max_ht - node.history_points_at(t)).clip(0)

    _, std = env.get_standardize_params(state[node.type], node.type)
    std[0:2] = env.attention_radius[(node.type, node.type)]
    rel_state = np.zeros_like(x[0])
    rel_state[0:2] = np.array(x)[-1, 0:2]
    x_st = env.standardize(x, state[node.type], node.type, mean=rel_state, std=std)
    if list(pred_state[node.type].keys())[0] == 'position':  # If we predict position we do it relative to current pos
        y_st = env.standardize(y, pred_state[node.type], node.type, mean=rel_state[0:2])
    else:
        y_st = env.standardize(y, pred_state[node.type], node.type)

    x_t = torch.tensor(x, dtype=torch.float)
    y_t = torch.tensor(y, dtype=torch.float)
    #--------------ADDED--------------
    x_unf_t = torch.tensor(unf_x, dtype=torch.float)
    #---------------------------------   
    x_st_t = torch.tensor(x_st, dtype=torch.float)
    y_st_t = torch.tensor(y_st, dtype=torch.float)

    # Neighbors
    neighbors_data_st = None
    neighbors_edge_value = None
    if hyperparams['edge_encoding']:
        # Scene Graph
        scene_graph = scene.get_scene_graph(t,
                                            env.attention_radius,
                                            hyperparams['edge_addition_filter'],
                                            hyperparams['edge_removal_filter']) if scene_graph is None else scene_graph

        neighbors_data_st = dict()
        neighbors_edge_value = dict()
        for edge_type in edge_types:
            neighbors_data_st[edge_type] = list()
            # We get all nodes which are connected to the current node for the current timestep
            connected_nodes = scene_graph.get_neighbors(node, edge_type[1])

            if hyperparams['dynamic_edges'] == 'yes':
                # We get the edge masks for the current node at the current timestep
                edge_masks = torch.tensor(scene_graph.get_edge_scaling(node), dtype=torch.float)
                neighbors_edge_value[edge_type] = edge_masks

            for connected_node in connected_nodes:
                neighbor_state_np = connected_node.get(np.array([t - max_ht, t]),
                                                       state[connected_node.type],
                                                       padding=0.0)

                # Make State relative to node where neighbor and node have same state
                _, std = env.get_standardize_params(state[connected_node.type], node_type=connected_node.type)
                std[0:2] = env.attention_radius[edge_type]
                equal_dims = np.min((neighbor_state_np.shape[-1], x.shape[-1]))
                rel_state = np.zeros_like(neighbor_state_np)
                rel_state[:, ..., :equal_dims] = x[-1, ..., :equal_dims]
                neighbor_state_np_st = env.standardize(neighbor_state_np,
                                                       state[connected_node.type],
                                                       node_type=connected_node.type,
                                                       mean=rel_state,
                                                       std=std)

                neighbor_state = torch.tensor(neighbor_state_np_st, dtype=torch.float)
                neighbors_data_st[edge_type].append(neighbor_state)

    # Robot
    robot_traj_st_t = None
    if hyperparams['incl_robot_node']:
        timestep_range_r = np.array([t, t + max_ft])
        if scene.non_aug_scene is not None:
            robot = scene.get_node_by_id(scene.non_aug_scene.robot.id)
        else:
            robot = scene.robot
        robot_type = robot.type
        robot_traj = robot.get(timestep_range_r, state[robot_type], padding=0.0)
        node_state = np.zeros_like(robot_traj[0])
        node_state[:x.shape[1]] = x[-1]
        robot_traj_st_t = get_relative_robot_traj(env, state, node_state, robot_traj, node.type, robot_type)

    # Map
    map_tuple = None
    viz_map_tuple = None
    if hyperparams['use_map_encoding']:
        if node.type in hyperparams['map_encoder']:
            if node.non_aug_node is not None:
                x = node.non_aug_node.get(np.array([t]), state[node.type])
            me_hyp = hyperparams['map_encoder'][node.type]
            if 'heading_state_index' in me_hyp:
                heading_state_index = me_hyp['heading_state_index']
                # We have to rotate the map in the opposit direction of the agent to match them
                if type(heading_state_index) is list:  # infer from velocity or heading vector
                    heading_angle = -np.arctan2(x[-1, heading_state_index[1]],
                                                x[-1, heading_state_index[0]]) * 180 / np.pi
                else:
                    heading_angle = -x[-1, heading_state_index] * 180 / np.pi
            else:
                heading_angle = None

            scene_map = scene.map[node.type]
            viz_scene_map = scene.map['VISUALIZATION']
            map_point = x[-1, :2]


            patch_size = hyperparams['map_encoder'][node.type]['patch_size']
            map_tuple = (scene_map, map_point, heading_angle, patch_size)
            viz_map_tuple = (viz_scene_map, map_point, heading_angle, [ps*3 for ps in patch_size])
            #--------------ADDED--------------
    map_name = scene.map_name
    if map_name == 'boston-seaport':
        map_num = 1
    elif map_name == 'singapore-onenorth':
        map_num = 2
    elif map_name == 'singapore-queenstown':
        map_num = 3
    elif map_name == 'singapore-hollandvillage':
        map_num = 4
    else:
        map_num = 0
    
    map_num_t = torch.tensor(map_num, dtype=torch.float)

    # # calculate cluster for this scene
    # viz_map = scene.map['VISUALIZATION']
    # viz_map = viz_map.as_image()
    # extent = [-viz_map.shape[1]/2., viz_map.shape[1]/2., -viz_map.shape[0]/2., viz_map.shape[0]/2. ]
    # plt.imshow(viz_map, origin='lower', extent=extent, alpha=0.5)
    # plt.axis('off')
    # plt.savefig('media/images{}/autobot/temp.png'.format(suffix), 
    #             dpi=200, bbox_inches='tight', pad_inches=0)

    # # get embeddings
    # xemb, yemb = get_embeddings() #runs model_colormaponly.pth
    # xemb = np.concatenate(xemb, 0)

    # # calculate cluster_assignment
    # pca_embedding = PCA.transform(xemb)
    # label = GM.predict(pca_embedding) 
    # label = torch.tensor(label[0])
    # label=0

    return (first_history_index, x_t, y_t, x_st_t, y_st_t, neighbors_data_st,
            neighbors_edge_value, robot_traj_st_t, map_tuple, viz_map_tuple,
            x_unf_t, map_num_t)
            #---------------------------------


def get_timesteps_data(env, scene, t, node_type, state, pred_state,
                       edge_types, min_ht, max_ht, min_ft, max_ft, hyperparams):
    """
    Puts together the inputs for ALL nodes in a given scene and timestep in it.

    :param env: Environment
    :param scene: Scene
    :param t: Timestep in scene
    :param node_type: Node Type of nodes for which the data shall be pre-processed
    :param state: Specification of the node state
    :param pred_state: Specification of the prediction state
    :param edge_types: List of all Edge Types for which neighbors are pre-processed
    :param max_ht: Maximum history timesteps
    :param max_ft: Maximum future timesteps (prediction horizon)
    :param hyperparams: Model hyperparameters
    :return:
    """
    nodes_per_ts = scene.present_nodes(t,
                                       type=node_type,
                                       min_history_timesteps=min_ht,
                                       min_future_timesteps=max_ft,
                                       return_robot=not hyperparams['incl_robot_node'])
    batch = list()
    nodes = list()
    out_timesteps = list()
    for timestep in nodes_per_ts.keys():
            scene_graph = scene.get_scene_graph(timestep,
                                                env.attention_radius,
                                                hyperparams['edge_addition_filter'],
                                                hyperparams['edge_removal_filter'])
            present_nodes = nodes_per_ts[timestep]
            for node in present_nodes:
                nodes.append(node)
                out_timesteps.append(timestep)
                batch.append(get_node_timestep_data(env, scene, timestep, node, state, pred_state,
                                                    edge_types, max_ht, max_ft, hyperparams,
                                                    scene_graph=scene_graph))
    if len(out_timesteps) == 0:
        return None
    return collate(batch), nodes, out_timesteps
