import warnings
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
sys.path.append("Trajectron-plus-plus/trajectron/")
from model.mgcvae import MultimodalGenerativeCVAE
from model.components import *
from model.model_utils import *
import model.dynamics as dynamic_module
from environment.scene_graph import DirectedEdge
from mgcvae_risk import multiply_gmms, add_gmm_params
# from ensemble_params import NUM_ENSEMBLE, activation_func

class MultimodalGenerativeCVAEEncoder(MultimodalGenerativeCVAE):
    def __init__(self,
                 env,
                 node_type,
                 ens_index,
                 model_registrar,
                 hyperparams,
                 device,
                 edge_types,
                 log_writer=None):
        self.hyperparams = hyperparams
        self.env = env
        self.node_type_str = node_type
        self.node_type = node_type + str(ens_index)
        self.model_registrar = model_registrar
        self.log_writer = log_writer
        self.device = device
        self.edge_types = [edge_type for edge_type in edge_types if edge_type[0] is node_type]
        self.curr_iter = 0
        self.node_modules = dict()
        self.min_hl = self.hyperparams['minimum_history_length']
        self.max_hl = self.hyperparams['maximum_history_length']
        self.ph = self.hyperparams['prediction_horizon']
        self.state = self.hyperparams['state']
        self.pred_state = self.hyperparams['pred_state'][node_type]
        self.state_length = int(np.sum([len(entity_dims) for entity_dims in self.state[node_type].values()]))
        if self.hyperparams['incl_robot_node']:
            self.robot_state_length = int(
                np.sum([len(entity_dims) for entity_dims in self.state[env.robot_type].values()])
            )
        self.pred_state_length = int(np.sum([len(entity_dims) for entity_dims in self.pred_state.values()]))
        edge_types_str = [DirectedEdge.get_str_from_types(*edge_type) for edge_type in self.edge_types]
        self.create_graphical_model(edge_types_str)
        dynamic_class = getattr(dynamic_module, hyperparams['dynamic'][self.node_type_str]['name'])
        dyn_limits = hyperparams['dynamic'][self.node_type_str]['limits']
        self.dynamic = dynamic_class(self.env.scenes[0].dt, dyn_limits, device,
                                     self.model_registrar, self.x_size, self.node_type_str)
    def create_node_models(self):
        ############################
        #   Node History Encoder   #
        ############################
        self.add_submodule(self.node_type + '/node_history_encoder',
                           model_if_absent=nn.LSTM(input_size=self.state_length,
                                                   hidden_size=self.hyperparams['enc_rnn_dim_history'],
                                                   batch_first=True))
        ###########################
        #   Node Future Encoder   #
        ###########################
        # We'll create this here, but then later check if in training mode.
        # Based on that, we'll factor this into the computation graph (or not).
        self.add_submodule(self.node_type + '/node_future_encoder',
                           model_if_absent=nn.LSTM(input_size=self.pred_state_length,
                                                   hidden_size=self.hyperparams['enc_rnn_dim_future'],
                                                   bidirectional=True,
                                                   batch_first=True))
        # These are related to how you initialize states for the node future encoder.
        self.add_submodule(self.node_type + '/node_future_encoder/initial_h',
                           model_if_absent=nn.Linear(self.state_length,
                                                     self.hyperparams['enc_rnn_dim_future']))
        self.add_submodule(self.node_type + '/node_future_encoder/initial_c',
                           model_if_absent=nn.Linear(self.state_length,
                                                     self.hyperparams['enc_rnn_dim_future']))
        ############################
        #   Robot Future Encoder   #
        ############################
        # We'll create this here, but then later check if we're next to the robot.
        # Based on that, we'll factor this into the computation graph (or not).
        if self.hyperparams['incl_robot_node']:
            self.add_submodule('robot_future_encoder',
                               model_if_absent=nn.LSTM(input_size=self.robot_state_length,
                                                       hidden_size=self.hyperparams['enc_rnn_dim_future'],
                                                       bidirectional=True,
                                                       batch_first=True))
            # These are related to how you initialize states for the robot future encoder.
            self.add_submodule('robot_future_encoder/initial_h',
                               model_if_absent=nn.Linear(self.robot_state_length,
                                                         self.hyperparams['enc_rnn_dim_future']))
            self.add_submodule('robot_future_encoder/initial_c',
                               model_if_absent=nn.Linear(self.robot_state_length,
                                                         self.hyperparams['enc_rnn_dim_future']))
        if self.hyperparams['edge_encoding']:
            ##############################
            #   Edge Influence Encoder   #
            ##############################
            # NOTE: The edge influence encoding happens during calls
            # to forward or incremental_forward, so we don't create
            # a model for it here for the max and sum variants.
            if self.hyperparams['edge_influence_combine_method'] == 'bi-rnn':
                self.add_submodule(self.node_type + '/edge_influence_encoder',
                                   model_if_absent=nn.LSTM(input_size=self.hyperparams['enc_rnn_dim_edge'],
                                                           hidden_size=self.hyperparams['enc_rnn_dim_edge_influence'],
                                                           bidirectional=True,
                                                           batch_first=True))
                # Four times because we're trying to mimic a bi-directional
                # LSTM's output (which, here, is c and h from both ends).
                self.eie_output_dims = 4 * self.hyperparams['enc_rnn_dim_edge_influence']
            elif self.hyperparams['edge_influence_combine_method'] == 'attention':
                # Chose additive attention because of https://arxiv.org/pdf/1703.03906.pdf
                # We calculate an attention context vector using the encoded edges as the "encoder"
                # (that we attend _over_)
                # and the node history encoder representation as the "decoder state" (that we attend _on_).
                self.add_submodule(self.node_type + '/edge_influence_encoder',
                                   model_if_absent=AdditiveAttention(
                                       encoder_hidden_state_dim=self.hyperparams['enc_rnn_dim_edge_influence'],
                                       decoder_hidden_state_dim=self.hyperparams['enc_rnn_dim_history']))
                self.eie_output_dims = self.hyperparams['enc_rnn_dim_edge_influence']
        ###################
        #   Map Encoder   #
        ###################
        if self.hyperparams['use_map_encoding']:
            if self.node_type_str in self.hyperparams['map_encoder']:
                me_params = self.hyperparams['map_encoder'][self.node_type_str]
                self.add_submodule(self.node_type + '/map_encoder',
                                   model_if_absent=CNNMapEncoder(me_params['map_channels'],
                                                                 me_params['hidden_channels'],
                                                                 me_params['output_size'],
                                                                 me_params['masks'],
                                                                 me_params['strides'],
                                                                 me_params['patch_size']))
        ################################
        #   Discrete Latent Variable   #
        ################################
        self.latent = DiscreteLatent(self.hyperparams, self.device)
        ######################################################################
        #   Various Fully-Connected Layers from Encoder to Latent Variable   #
        ######################################################################
        # Node History Encoder
        x_size = self.hyperparams['enc_rnn_dim_history']
        if self.hyperparams['edge_encoding']:
            #              Edge Encoder
            x_size += self.eie_output_dims
        if self.hyperparams['incl_robot_node']:
            #              Future Conditional Encoder
            x_size += 4 * self.hyperparams['enc_rnn_dim_future']
        if self.hyperparams['use_map_encoding'] and self.node_type_str in self.hyperparams['map_encoder']:
            #              Map Encoder
            x_size += self.hyperparams['map_encoder'][self.node_type_str]['output_size']
        z_size = self.hyperparams['N'] * self.hyperparams['K']
        if self.hyperparams['p_z_x_MLP_dims'] is not None:
            self.add_submodule(self.node_type + '/p_z_x',
                               model_if_absent=nn.Linear(x_size, self.hyperparams['p_z_x_MLP_dims']))
            hx_size = self.hyperparams['p_z_x_MLP_dims']
        else:
            hx_size = x_size
        self.add_submodule(self.node_type + '/hx_to_z',
                           model_if_absent=nn.Linear(hx_size, self.latent.z_dim))
        if self.hyperparams['q_z_xy_MLP_dims'] is not None:
            self.add_submodule(self.node_type + '/q_z_xy',
                               #                                           Node Future Encoder
                               model_if_absent=nn.Linear(x_size + 4 * self.hyperparams['enc_rnn_dim_future'],
                                                         self.hyperparams['q_z_xy_MLP_dims']))
            hxy_size = self.hyperparams['q_z_xy_MLP_dims']
        else:
            #                           Node Future Encoder
            hxy_size = x_size + 4 * self.hyperparams['enc_rnn_dim_future']
        self.add_submodule(self.node_type + '/hxy_to_z',
                           model_if_absent=nn.Linear(hxy_size, self.latent.z_dim))
        self.x_size = x_size
        self.z_size = z_size
    def obtain_encoded_tensors(self,
                               mode,
                               inputs,
                               inputs_st,
                               labels,
                               labels_st,
                               first_history_indices,
                               neighbors,
                               neighbors_edge_value,
                               robot,
                               map) -> (torch.Tensor,
                                        torch.Tensor,
                                        torch.Tensor,
                                        torch.Tensor,
                                        torch.Tensor,
                                        torch.Tensor):
        """
        Encodes input and output tensors for node and robot.
        :param mode: Mode in which the model is operated. E.g. Train, Eval, Predict.
        :param inputs: Input tensor including the state for each agent over time [bs, t, state].
        :param inputs_st: Standardized input tensor.
        :param labels: Label tensor including the label output for each agent over time [bs, t, pred_state].
        :param labels_st: Standardized label tensor.
        :param first_history_indices: First timestep (index) in scene for which data is available for a node [bs]
        :param neighbors: Preprocessed dict (indexed by edge type) of list of neighbor states over time.
                            [[bs, t, neighbor state]]
        :param neighbors_edge_value: Preprocessed edge values for all neighbor nodes [[N]]
        :param robot: Standardized robot state over time. [bs, t, robot_state]
        :param map: Tensor of Map information. [bs, channels, x, y]
        :return: tuple(x, x_nr_t, y_e, y_r, y, n_s_t0)
            WHERE
            - x: Encoded input / condition tensor to the CVAE x_e.
            - x_r_t: Robot state (if robot is in scene).
            - y_e: Encoded label / future of the node.
            - y_r: Encoded future of the robot.
            - y: Label / future of the node.
            - n_s_t0: Standardized current state of the node.
        """
        x, x_r_t, y_e, y_r, y = None, None, None, None, None
        initial_dynamics = dict()
        batch_size = inputs.shape[0]
        #########################################
        # Provide basic information to encoders #
        #########################################
        node_history = inputs
        node_present_state = inputs[:, -1]
        node_pos = inputs[:, -1, 0:2]
        node_vel = inputs[:, -1, 2:4]
        node_history_st = inputs_st
        node_present_state_st = inputs_st[:, -1]
        node_pos_st = inputs_st[:, -1, 0:2]
        node_vel_st = inputs_st[:, -1, 2:4]
        n_s_t0 = node_present_state_st
        initial_dynamics['pos'] = node_pos
        initial_dynamics['vel'] = node_vel
        self.dynamic.set_initial_condition(initial_dynamics)
        if self.hyperparams['incl_robot_node']:
            x_r_t, y_r = robot[..., 0, :], robot[..., 1:, :]
        ##################
        # Encode History #
        ##################
        node_history_encoded = self.encode_node_history(mode,
                                                        node_history_st,
                                                        first_history_indices)
        ##################
        # Encode Present #
        ##################
        node_present = node_present_state_st  # [bs, state_dim]
        ##################
        # Encode Future #
        ##################
        if mode != ModeKeys.PREDICT:
            y = labels_st
        ##############################
        # Encode Node Edges per Type #
        ##############################
        if self.hyperparams['edge_encoding']:
            node_edges_encoded = list()
            for edge_type in self.edge_types:
                # Encode edges for given edge type
                encoded_edges_type = self.encode_edge(mode,
                                                      node_history,
                                                      node_history_st,
                                                      edge_type,
                                                      neighbors[edge_type],
                                                      neighbors_edge_value[edge_type],
                                                      first_history_indices)
                node_edges_encoded.append(encoded_edges_type)  # List of [bs/nbs, enc_rnn_dim]
            #####################
            # Encode Node Edges #
            #####################
            total_edge_influence = self.encode_total_edge_influence(mode,
                                                                    node_edges_encoded,
                                                                    node_history_encoded,
                                                                    batch_size)
        ################
        # Map Encoding #
        ################
        if self.hyperparams['use_map_encoding'] and self.node_type_str in self.hyperparams['map_encoder']:
            if self.log_writer and (self.curr_iter + 1) % 500 == 0:
                map_clone = map.clone()
                map_patch = self.hyperparams['map_encoder'][self.node_type_str]['patch_size']
                map_clone[:, :, map_patch[1] - 5:map_patch[1] + 5, map_patch[0] - 5:map_patch[0] + 5] = 1.
                self.log_writer.add_images(f"{self.node_type}/cropped_maps", map_clone,
                                           self.curr_iter, dataformats='NCWH')
            encoded_map = self.node_modules[self.node_type + '/map_encoder'](map * 2. - 1., (mode == ModeKeys.TRAIN))
            do = self.hyperparams['map_encoder'][self.node_type_str]['dropout']
            encoded_map = F.dropout(encoded_map, do, training=(mode == ModeKeys.TRAIN))
        ######################################
        # Concatenate Encoder Outputs into x #
        ######################################
        x_concat_list = list()
        # Every node has an edge-influence encoder (which could just be zero).
        if self.hyperparams['edge_encoding']:
            x_concat_list.append(total_edge_influence)  # [bs/nbs, 4*enc_rnn_dim]
        # Every node has a history encoder.
        x_concat_list.append(node_history_encoded)  # [bs/nbs, enc_rnn_dim_history]
        if self.hyperparams['incl_robot_node']:
            robot_future_encoder = self.encode_robot_future(mode, x_r_t, y_r)
            x_concat_list.append(robot_future_encoder)
        if self.hyperparams['use_map_encoding'] and self.node_type_str in self.hyperparams['map_encoder']:
            if self.log_writer:
                self.log_writer.add_scalar(f"{self.node_type}/encoded_map_max",
                                           torch.max(torch.abs(encoded_map)), self.curr_iter)
            x_concat_list.append(encoded_map)
        x = torch.cat(x_concat_list, dim=1)
        if mode == ModeKeys.TRAIN or mode == ModeKeys.EVAL:
            y_e = self.encode_node_future(mode, node_present, y)
        return x, x_r_t, y_e, y_r, y, n_s_t0

    def train_loss_encode(self,
                   inputs,
                   inputs_st,
                   first_history_indices,
                   labels,
                   labels_st,
                   neighbors,
                   neighbors_edge_value,
                   robot,
                   map,
                   prediction_horizon,
                   # -------------- ADDED ----------------
                   heatmap_tensor,
                   x_unf,
                   map_name,
                   grid_tensor
                   # -------------------------------------                   
                   ) -> torch.Tensor:
        """
        Calculates the training loss for a batch.

        :param inputs: Input tensor including the state for each agent over time [bs, t, state].
        :param inputs_st: Standardized input tensor.
        :param first_history_indices: First timestep (index) in scene for which data is available for a node [bs]
        :param labels: Label tensor including the label output for each agent over time [bs, t, pred_state].
        :param labels_st: Standardized label tensor.
        :param neighbors: Preprocessed dict (indexed by edge type) of list of neighbor states over time.
                            [[bs, t, neighbor state]]
        :param neighbors_edge_value: Preprocessed edge values for all neighbor nodes [[N]]
        :param robot: Standardized robot state over time. [bs, t, robot_state]
        :param map: Tensor of Map information. [bs, channels, x, y]
        :param prediction_horizon: Number of prediction timesteps.
        :return: Scalar tensor -> nll loss
        """
        mode = ModeKeys.TRAIN

        x, x_nr_t, y_e, y_r, y, n_s_t0 = self.obtain_encoded_tensors(mode=mode,
                                                                     inputs=inputs,
                                                                     inputs_st=inputs_st,
                                                                     labels=labels,
                                                                     labels_st=labels_st,
                                                                     first_history_indices=first_history_indices,
                                                                     neighbors=neighbors,
                                                                     neighbors_edge_value=neighbors_edge_value,
                                                                     robot=robot,
                                                                     map=map)
        z, kl = self.encoder(mode, x, y_e)
        return z, kl, (x, x_nr_t, y_e, y_r, y, n_s_t0), (self.latent, self.dynamic)
    
    def eval_loss_encode(self,
                  inputs,
                  inputs_st,
                  first_history_indices,
                  labels,
                  labels_st,
                  neighbors,
                  neighbors_edge_value,
                  robot,
                  map,
                  prediction_horizon) -> torch.Tensor:
        """
        Calculates the evaluation loss for a batch.

        :param inputs: Input tensor including the state for each agent over time [bs, t, state].
        :param inputs_st: Standardized input tensor.
        :param first_history_indices: First timestep (index) in scene for which data is available for a node [bs]
        :param labels: Label tensor including the label output for each agent over time [bs, t, pred_state].
        :param labels_st: Standardized label tensor.
        :param neighbors: Preprocessed dict (indexed by edge type) of list of neighbor states over time.
                            [[bs, t, neighbor state]]
        :param neighbors_edge_value: Preprocessed edge values for all neighbor nodes [[N]]
        :param robot: Standardized robot state over time. [bs, t, robot_state]
        :param map: Tensor of Map information. [bs, channels, x, y]
        :param prediction_horizon: Number of prediction timesteps.
        :return: tuple(nll_q_is, nll_p, nll_exact, nll_sampled)
        """

        mode = ModeKeys.EVAL

        x, x_nr_t, y_e, y_r, y, n_s_t0 = self.obtain_encoded_tensors(mode=mode,
                                                                     inputs=inputs,
                                                                     inputs_st=inputs_st,
                                                                     labels=labels,
                                                                     labels_st=labels_st,
                                                                     first_history_indices=first_history_indices,
                                                                     neighbors=neighbors,
                                                                     neighbors_edge_value=neighbors_edge_value,
                                                                     robot=robot,
                                                                     map=map)

        ### Importance sampled NLL estimate
        z, _ = self.encoder(mode, x, y_e)  # [k_eval, nbs, N*K]
        return z, (x, x_nr_t, y_e, y_r, y, n_s_t0), (self.latent, self.dynamic)

    def predict_encode(self,
                inputs,
                inputs_st,
                first_history_indices,
                neighbors,
                neighbors_edge_value,
                robot,
                map,
                prediction_horizon,
                num_samples,
                z_mode=False,
                gmm_mode=False,
                full_dist=True,
                all_z_sep=False,
                encoder_output = None):
        """
        Predicts the future of a batch of nodes.

        :param inputs: Input tensor including the state for each agent over time [bs, t, state].
        :param inputs_st: Standardized input tensor.
        :param first_history_indices: First timestep (index) in scene for which data is available for a node [bs]
        :param neighbors: Preprocessed dict (indexed by edge type) of list of neighbor states over time.
                            [[bs, t, neighbor state]]
        :param neighbors_edge_value: Preprocessed edge values for all neighbor nodes [[N]]
        :param robot: Standardized robot state over time. [bs, t, robot_state]
        :param map: Tensor of Map information. [bs, channels, x, y]
        :param prediction_horizon: Number of prediction timesteps.
        :param num_samples: Number of samples from the latent space.
        :param z_mode: If True: Select the most likely latent state.
        :param gmm_mode: If True: The mode of the GMM is sampled.
        :param all_z_sep: Samples each latent mode individually without merging them into a GMM.
        :param full_dist: Samples all latent states and merges them into a GMM as output.
        :return:
        """
        mode = ModeKeys.PREDICT
        if encoder_output is None:
            x, x_nr_t, _, y_r, _, n_s_t0 = self.obtain_encoded_tensors(mode=mode,
                                                                    inputs=inputs,
                                                                    inputs_st=inputs_st,
                                                                    labels=None,
                                                                    labels_st=None,
                                                                    first_history_indices=first_history_indices,
                                                                    neighbors=neighbors,
                                                                    neighbors_edge_value=neighbors_edge_value,
                                                                    robot=robot,
                                                                    map=map)
            self.latent.p_dist = self.p_z_x(mode, x)
            z, num_samples, num_components = self.latent.sample_p(num_samples,
                                                                mode,
                                                                most_likely_z=z_mode,
                                                                full_dist=full_dist,
                                                                all_z_sep=all_z_sep)
            encoder_output = (x, x_nr_t, y_r, n_s_t0, num_samples, num_components, z)
        else:
            (x, x_nr_t, y_r, n_s_t0, num_samples, num_components, z) = encoder_output
        return encoder_output