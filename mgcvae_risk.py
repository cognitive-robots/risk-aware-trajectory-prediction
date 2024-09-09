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
# from ensemble_params import NUM_ENSEMBLE, activation_func

def reshape_to_components(tensor, components, dimensions):
    if len(tensor.shape) == 5:
        return tensor
    return torch.reshape(tensor, list(tensor.shape[:-1]) + [components, dimensions])

def to_variance(log_sigmas, corrs):
    sigma_x = torch.exp(log_sigmas)[:,:,:,0]
    sigma_y = torch.exp(log_sigmas)[:,:,:,1]
    rho_sigmax_sigmay = corrs * sigma_x * sigma_y

    var_left = torch.stack([sigma_x*sigma_x, rho_sigmax_sigmay], dim=-1)
    var_right = torch.stack([rho_sigmax_sigmay, sigma_y*sigma_y], dim=-1)

    variance = torch.stack([var_left, var_right], dim=-1)
    return variance

def from_variance(variance):
    sigma_x = torch.sqrt(variance[:,:,:,0,0]) + 1e-4 # to make sure there's no division by 0
    sigma_y = torch.sqrt(variance[:,:,:,1,1]) + 1e-4 # [256]
    if torch.any(sigma_x <= 0) or torch.any(sigma_x <= 0):
        import pdb; pdb.set_trace()

    rho_sigmax_sigmay = variance[:,:,:,1,0]
    rho = rho_sigmax_sigmay / sigma_x / sigma_y

    sigmas = torch.cat((sigma_x, sigma_y), dim=-1)
    log_sigmas = torch.log(sigmas)

    return(log_sigmas, rho)

def inverse(matrix): # keep getting nans with torch.inverse, this only works for 2D matrix
    det = torch.det(matrix)

def multiply_gmms(gmm1, gmm2):
    #################################################################
    # Following formula from https://math.stackexchange.com/a/2414813
    # Additionally, multiplying pis, and normalizing to get sum=1
    #################################################################

    (log_pis1, mus1, log_sigmas1, corrs1) = gmm1    
    (log_pis2, mus2, log_sigmas2, corrs2) = gmm2
    corrs1 = torch.clamp(corrs1, min=1e-5, max=(1.0 - 1e-5))
    corrs2 = torch.clamp(corrs2, min=1e-5, max=(1.0 - 1e-5))
    components = log_pis1.shape[-1]

    # multiply pis and normalize
    pis1 = torch.exp(log_pis1)
    pis2 = torch.exp(log_pis2)
    pis3 = pis1 * pis2
    log_pis3 = log_pis1 + log_pis2 - torch.log(torch.sum(pis3, dim=-1, keepdims=True))

    # multiply GMMs
    mu1 = reshape_to_components(mus1, components, 2).unsqueeze(dim=-1)
    mu2 = reshape_to_components(mus2, components, 2).unsqueeze(dim=-1)

    log_sigma1 = reshape_to_components(log_sigmas1, components, 2)
    log_sigma2 = reshape_to_components(log_sigmas2, components, 2)    
    variance1 = to_variance(log_sigma1, corrs1)
    variance2 = to_variance(log_sigma2, corrs2)

    # # Calculations using Cholesky decomposition, which don't work (give non-symmetric variances sometimes)
    # L = torch.cholesky(variance1 + variance2)
    # L_inv = torch.inverse(L)
    # mu1_tilde = torch.matmul(L_inv, mu1)
    # mu2_tilde = torch.matmul(L_inv, mu2)
    # variance1_tilde = torch.matmul(L_inv, variance1)
    # variance2_tilde = torch.matmul(L_inv, variance2)
    # variance1_tilde_T = torch.transpose(variance1_tilde, dim0=-2, dim1=-1)
    # variance2_tilde_T = torch.transpose(variance2_tilde, dim0=-2, dim1=-1)
    # variance3 = torch.matmul(variance1_tilde_T, variance2_tilde)
    # mu3 = torch.matmul(variance2_tilde_T, mu1_tilde) + torch.matmul(variance1_tilde_T, mu2_tilde)

    # Calculations using 3 inverse calculations (much more expensive, but doesn't give non-symmetric variances)
    variance1_inv = torch.inverse(variance1)
    variance2_inv = torch.inverse(variance2)
    variance3 = torch.inverse(variance1_inv + variance2_inv)
    mu3 = torch.matmul(variance3, torch.matmul(variance1_inv, mu1)) + torch.matmul(variance3, torch.matmul(variance2_inv, mu2))

    mus3 = torch.cat((mu3[:,:,:,0,0], mu3[:,:,:,1,0]), dim=-1)
    (log_sigmas3, corrs3) = from_variance(variance3)
    corrs3 = torch.clamp(corrs3, min=1e-5, max=(1.0 - 1e-5))

    # checking for nans or invalid correlation values
    indices = torch.cat((torch.nonzero(~torch.isfinite(corrs3)), torch.nonzero(corrs3*corrs3 >= 1.0)))
    for index in indices:
        import pdb; pdb.set_trace()
        symm_matrix = (variance1_inv + variance2_inv)[index[0], index[1], index[2]]
        det = torch.det(symm_matrix)
        b = symm_matrix[0,1]
        new_b = -b / det
        variance3[index[0], index[1], index[2], 0, 1] = new_b
        variance3[index[0], index[1], index[2], 1, 0] = new_b
    (log_sigmas3, corrs3) = from_variance(variance3)

    gmm3 = (log_pis3, mus3, log_sigmas3, corrs3)
    for e in gmm3:
        if torch.any(~torch.isfinite(e)):
            import pdb; pdb.set_trace()
    return gmm3

def add_gmm_params(gmm1, gmm2, lamda=1): # gmm1 is the prev combined params, gmm2 is the new one
    (log_pis1, mus1, log_sigmas1, corrs1) = gmm1    
    (log_pis2, mus2, log_sigmas2, corrs2) = gmm2

    pis1 = torch.exp(log_pis1)
    pis2 = torch.exp(log_pis2)
    pis3 = pis1 + lamda * pis2 / (1 + lamda)
    log_pis3 = torch.log(pis3)

    mus3 = mus1 + lamda * mus2

    log_sigmas3 = log_sigmas1 + lamda * log_sigmas2

    corrs3 = corrs1 + lamda * corrs2
    corrs3 = torch.clamp(corrs3, min=1e-5, max=(1.0 - 1e-5))

    gmm3 = (log_pis3, mus3, log_sigmas3, corrs3)
    for e in gmm3:
        if torch.any(~torch.isfinite(e)):
            import pdb; pdb.set_trace()
    return gmm3

def both_gmm_params(gmm1, gmm2): # gmm1 is the prev combined params, gmm2 is the new one
    (log_pis1, mus1, log_sigmas1, corrs1) = gmm1    
    (log_pis2, mus2, log_sigmas2, corrs2) = gmm2
    log_pis3 = torch.cat([log_pis1 / 2, log_pis2 / 2], dim=-1)
    mus3 = torch.cat([mus1, mus2], dim=-1)
    log_sigmas3 = torch.cat([log_sigmas1, log_sigmas2], dim=-1)
    corrs3 = torch.cat([corrs1, corrs2], dim=-1)
    gmm3 = (log_pis3, mus3, log_sigmas3, corrs3)
    return gmm3

class MultimodalGenerativeCVAERisk(MultimodalGenerativeCVAE):
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
                                     self.model_registrar, self.x_size, self.node_type)


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
            if self.node_type in self.hyperparams['map_encoder']:
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
        if self.hyperparams['use_map_encoding'] and self.node_type in self.hyperparams['map_encoder']:
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

        ####################
        #   Decoder LSTM   #
        ####################
        if self.hyperparams['incl_robot_node']:
            decoder_input_dims = self.pred_state_length + self.robot_state_length + z_size + x_size
        else:
            decoder_input_dims = self.pred_state_length + z_size + x_size

        self.add_submodule(self.node_type + '/decoder/state_action',
                           model_if_absent=nn.Sequential(
                               nn.Linear(self.state_length, self.pred_state_length)))

        self.add_submodule(self.node_type + '/decoder/rnn_cell',
                           model_if_absent=nn.GRUCell(decoder_input_dims, self.hyperparams['dec_rnn_dim']))
        self.add_submodule(self.node_type + '/decoder/initial_h',
                           model_if_absent=nn.Linear(z_size + x_size, self.hyperparams['dec_rnn_dim']))

        ###################
        #   Decoder GMM   #
        ###################
        self.add_submodule(self.node_type + '/decoder/proj_to_GMM_log_pis',
                           model_if_absent=nn.Linear(self.hyperparams['dec_rnn_dim'],
                                                     self.hyperparams['GMM_components']))
        self.add_submodule(self.node_type + '/decoder/proj_to_GMM_mus',
                           model_if_absent=nn.Linear(self.hyperparams['dec_rnn_dim'],
                                                     self.hyperparams['GMM_components'] * self.pred_state_length))
        self.add_submodule(self.node_type + '/decoder/proj_to_GMM_log_sigmas',
                           model_if_absent=nn.Linear(self.hyperparams['dec_rnn_dim'],
                                                     self.hyperparams['GMM_components'] * self.pred_state_length))
        self.add_submodule(self.node_type + '/decoder/proj_to_GMM_corrs',
                           model_if_absent=nn.Linear(self.hyperparams['dec_rnn_dim'],
                                                     self.hyperparams['GMM_components']))

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
        if self.hyperparams['use_map_encoding'] and self.node_type in self.hyperparams['map_encoder']:
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

        if self.hyperparams['use_map_encoding'] and self.node_type in self.hyperparams['map_encoder']:
            if self.log_writer:
                self.log_writer.add_scalar(f"{self.node_type}/encoded_map_max",
                                           torch.max(torch.abs(encoded_map)), self.curr_iter)
            x_concat_list.append(encoded_map)

        x = torch.cat(x_concat_list, dim=1)

        if mode == ModeKeys.TRAIN or mode == ModeKeys.EVAL:
            y_e = self.encode_node_future(mode, node_present, y)

        return x, x_r_t, y_e, y_r, y, n_s_t0

    # def create_graphical_model(self, edge_types):
    #     """
    #     Creates or queries all trainable components.

    #     :param edge_types: List containing strings for all possible edge types for the node type.
    #     :return: None
    #     """
    #     self.clear_submodules()

    #     ############################
    #     #   Everything but Edges   #
    #     ############################
    #     self.create_node_models()
    #     self.add_submodule(self.node_type + '/ensemble/aggregator',
    #                        model_if_absent=nn.Linear(x_size, len(NUM_ENSEMBLE)))
    #     #####################
    #     #   Edge Encoders   #
    #     #####################
    #     if self.hyperparams['edge_encoding']:
    #         self.create_edge_models(edge_types)

    #     for name, module in self.node_modules.items():
    #         module.to(self.device)

    def p_y_xz(self, mode, x, x_nr_t, y_r, n_s_t0, z_stacked, prediction_horizon, # to allow multiplying of gmms
               num_samples, num_components=1, gmm_mode=False):
        r"""
        .. math:: p_\psi(\mathbf{y}_i \mid \mathbf{x}_i, z)

        :param mode: Mode in which the model is operated. E.g. Train, Eval, Predict.
        :param x: Input / Condition tensor.
        :param x_nr_t: Joint state of node and robot (if robot is in scene).
        :param y: Future tensor.
        :param y_r: Encoded future tensor.
        :param n_s_t0: Standardized current state of the node.
        :param z_stacked: Stacked latent state. [num_samples_z * num_samples_gmm, bs, latent_state]
        :param prediction_horizon: Number of prediction timesteps.
        :param num_samples: Number of samples from the latent space.
        :param num_components: Number of GMM components.
        :param gmm_mode: If True: The mode of the GMM is sampled.
        :return: GMM2D. If mode is Predict, also samples from the GMM.
        """
        ph = prediction_horizon
        pred_dim = self.pred_state_length

        z = torch.reshape(z_stacked, (-1, self.latent.z_dim))
        zx = torch.cat([z, x.repeat(num_samples * num_components, 1)], dim=1)

        cell = self.node_modules[self.node_type + '/decoder/rnn_cell']
        initial_h_model = self.node_modules[self.node_type + '/decoder/initial_h']

        initial_state = initial_h_model(zx)

        log_pis, mus, log_sigmas, corrs, a_sample = [], [], [], [], []

        # Infer initial action state for node from current state
        a_0 = self.node_modules[self.node_type + '/decoder/state_action'](n_s_t0)

        state = initial_state
        if self.hyperparams['incl_robot_node']:
            input_ = torch.cat([zx,
                                a_0.repeat(num_samples * num_components, 1),
                                x_nr_t.repeat(num_samples * num_components, 1)], dim=1)
        else:
            input_ = torch.cat([zx, a_0.repeat(num_samples * num_components, 1)], dim=1)

        for j in range(ph):
            h_state = cell(input_, state)
            log_pi_t, mu_t, log_sigma_t, corr_t = self.project_to_GMM_params(h_state)

            gmm = GMM2D(log_pi_t, mu_t, log_sigma_t, corr_t)  # [k;bs, pred_dim]

            if mode == ModeKeys.PREDICT and gmm_mode:
                a_t = gmm.mode()
            else:
                a_t = gmm.rsample()

            if num_components > 1:
                if mode == ModeKeys.PREDICT:
                    log_pis.append(self.latent.p_dist.logits.repeat(num_samples, 1, 1))
                else:
                    log_pis.append(self.latent.q_dist.logits.repeat(num_samples, 1, 1))
            else:
                log_pis.append(
                    torch.ones_like(corr_t.reshape(num_samples, num_components, -1).permute(0, 2, 1).reshape(-1, 1))
                )

            mus.append(
                mu_t.reshape(
                    num_samples, num_components, -1, 2
                ).permute(0, 2, 1, 3).reshape(-1, 2 * num_components)
            )
            log_sigmas.append(
                log_sigma_t.reshape(
                    num_samples, num_components, -1, 2
                ).permute(0, 2, 1, 3).reshape(-1, 2 * num_components))
            corrs.append(
                corr_t.reshape(
                    num_samples, num_components, -1
                ).permute(0, 2, 1).reshape(-1, num_components))

            if self.hyperparams['incl_robot_node']:
                dec_inputs = [zx, a_t, y_r[:, j].repeat(num_samples * num_components, 1)]
            else:
                dec_inputs = [zx, a_t]
            input_ = torch.cat(dec_inputs, dim=1)
            state = h_state

        log_pis = torch.stack(log_pis, dim=1)
        mus = torch.stack(mus, dim=1)
        log_sigmas = torch.stack(log_sigmas, dim=1)
        corrs = torch.stack(corrs, dim=1)

        ### ----NEW PART ----
        if self.prev_gmm_params:
            curr_gmm_params = (log_pis, mus, log_sigmas, corrs)
            combined_gmm_params = multiply_gmms(self.prev_gmm_params, curr_gmm_params)
            (log_pis, mus, log_sigmas, corrs) = combined_gmm_params

        self.gmm_params = (log_pis.detach(), mus.detach(), log_sigmas.detach(), corrs.detach())
        # num_components = log_pis.shape[-1] # changes if we're doing both_gmm_params
        ###-------------------

        a_dist = GMM2D(torch.reshape(log_pis, [num_samples, -1, ph, num_components]),
                       torch.reshape(mus, [num_samples, -1, ph, num_components * pred_dim]),
                       torch.reshape(log_sigmas, [num_samples, -1, ph, num_components * pred_dim]),
                       torch.reshape(corrs, [num_samples, -1, ph, num_components]))

        if self.hyperparams['dynamic'][self.node_type_str]['distribution']:
            y_dist = self.dynamic.integrate_distribution(a_dist, x)
            if torch.any(~torch.isfinite(y_dist.corrs)) or torch.any(~torch.isfinite(y_dist.log_sigmas)):
                import pdb; pdb.set_trace()
                y_dist.corrs = torch.nan_to_num(y_dist.corrs)
                y_dist.log_sigmas = torch.nan_to_num(y_dist.log_sigmas)
                
        else:
            y_dist = a_dist

        if mode == ModeKeys.PREDICT:
            if gmm_mode:
                a_sample = a_dist.mode()
            else:
                a_sample = a_dist.rsample()
            sampled_future = self.dynamic.integrate_samples(a_sample, x)
            return y_dist, sampled_future
        else:
            return y_dist

    def train_loss(self,
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
        log_p_y_xz = self.decoder(mode, x, x_nr_t, y, y_r, n_s_t0, z,
                                  labels,  # Loss is calculated on unstandardized label
                                  prediction_horizon,
                                  self.hyperparams['k'])


        log_p_y_xz_mean = torch.mean(log_p_y_xz, dim=0)  # [nbs]
        # sum_of_weights = 0
        # for iter in range(len(map_name)):
        #     if map_name[iter] != 0: #map successful
        #         risk_weight = 1
        #         if loc_risk:
        #             ###RISKY LOCATION
        #             ##describing what each variable is:
        #             #grid_tensor=x_min,x_max,y_min,y_max,x_grid_size,y_grid_size: 
        #             #for maps boston-seaport, singapore-onenorth, singapore-queenstown, singapore-hollandvillage
                    
        #             ##Work out grid loc from unf_x
        #             map_number = map_name[iter].item()
        #             map_integer = int(map_number)
        #             x_val = (x_unf[iter][-1][0]-grid_tensor[0][map_integer-1])//grid_tensor[4][map_integer-1]
        #             y_val = (x_unf[iter][-1][1]-grid_tensor[2][map_integer-1])//grid_tensor[5][map_integer-1]

        #             # Bear in mind, this assumes a 100x100 grid 
        #             if y_val == 0:
        #                 y_val = 1
        #             if x_val == 100:
        #                 x_val = 99
        #             grid_loc = (100*(100-y_val)) + x_val
        #             grid_loc_int = int(grid_loc.item())
        #             if grid_loc_int > 10000:
        #                 log_p_y_xz_mean[iter] = log_p_y_xz_mean[iter] 
        #                 sum_of_weights = sum_of_weights + 1
        #             #Multiply Loss term
        #             else:
        #                 log_p_y_xz_mean[iter] = log_p_y_xz_mean[iter] * heatmap_tensor[grid_loc_int][map_integer]
        #                 sum_of_weights = sum_of_weights + heatmap_tensor[grid_loc_int][map_integer]
        #     else:
        #         log_p_y_xz_mean[iter] = log_p_y_xz_mean[iter] 
        #         sum_of_weights = sum_of_weights + 1

        log_likelihood = torch.mean(log_p_y_xz_mean)

        mutual_inf_q = mutual_inf_mc(self.latent.q_dist)
        mutual_inf_p = mutual_inf_mc(self.latent.p_dist)

        ELBO = log_likelihood - self.kl_weight * kl + 1. * mutual_inf_p
        loss = -ELBO

        if self.hyperparams['log_histograms'] and self.log_writer is not None:
            self.log_writer.add_histogram('%s/%s' % (str(self.node_type), 'log_p_y_xz'),
                                          log_p_y_xz_mean,
                                          self.curr_iter)

        if self.log_writer is not None:
            self.log_writer.add_scalar('%s/%s' % (str(self.node_type), 'mutual_information_q'),
                                       mutual_inf_q,
                                       self.curr_iter)
            self.log_writer.add_scalar('%s/%s' % (str(self.node_type), 'mutual_information_p'),
                                       mutual_inf_p,
                                       self.curr_iter)
            self.log_writer.add_scalar('%s/%s' % (str(self.node_type), 'log_likelihood'),
                                       log_likelihood,
                                       self.curr_iter)
            self.log_writer.add_scalar('%s/%s' % (str(self.node_type), 'loss'),
                                       loss,
                                       self.curr_iter)
            if self.hyperparams['log_histograms']:
                self.latent.summarize_for_tensorboard(self.log_writer, str(self.node_type), self.curr_iter)
        return loss

    def eval_loss(self,
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

        num_components = self.hyperparams['N'] * self.hyperparams['K']
        ### Importance sampled NLL estimate
        z, _ = self.encoder(mode, x, y_e)  # [k_eval, nbs, N*K]
        z = self.latent.sample_p(1, mode, full_dist=True)
        y_dist, _ = self.p_y_xz(ModeKeys.PREDICT, x, x_nr_t, y_r, n_s_t0, z,
                                prediction_horizon, num_samples=1, num_components=num_components)
        # We use unstandardized labels to compute the loss
        log_p_yt_xz = torch.clamp(y_dist.log_prob(labels), max=self.hyperparams['log_p_yt_xz_max'])
        log_p_y_xz = torch.sum(log_p_yt_xz, dim=2)
        log_p_y_xz_mean = torch.mean(log_p_y_xz, dim=0)  # [nbs]
        log_likelihood = torch.mean(log_p_y_xz_mean)
        nll = -log_likelihood
        return nll

    def predict(self,
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
                all_z_sep=False):
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

        _, our_sampled_future = self.p_y_xz(mode, x, x_nr_t, y_r, n_s_t0, z,
                                            prediction_horizon,
                                            num_samples,
                                            num_components,
                                            gmm_mode)

        return x, our_sampled_future

    def train_loss_pt1(self,
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
        log_p_y_xz = self.decoder(mode, x, x_nr_t, y, y_r, n_s_t0, z,
                                  labels,  # Loss is calculated on unstandardized label
                                  prediction_horizon,
                                  self.hyperparams['k'])


        log_p_y_xz_mean = torch.mean(log_p_y_xz, dim=0)  # [nbs]

        mutual_inf_q = mutual_inf_mc(self.latent.q_dist)
        mutual_inf_p = mutual_inf_mc(self.latent.p_dist)

        if self.hyperparams['log_histograms'] and self.log_writer is not None:
            self.log_writer.add_histogram('%s/%s' % (str(self.node_type), 'log_p_y_xz'),
                                          log_p_y_xz_mean,
                                          self.curr_iter)

        if self.log_writer is not None:
            self.log_writer.add_scalar('%s/%s' % (str(self.node_type), 'mutual_information_q'),
                                       mutual_inf_q,
                                       self.curr_iter)
            self.log_writer.add_scalar('%s/%s' % (str(self.node_type), 'mutual_information_p'),
                                       mutual_inf_p,
                                       self.curr_iter)
            if self.hyperparams['log_histograms']:
                self.latent.summarize_for_tensorboard(self.log_writer, str(self.node_type), self.curr_iter)
            
        kl_term = self.kl_weight * kl
        inf_term = 1. * mutual_inf_p
        return x, log_p_y_xz_mean, kl_term, inf_term

    def eval_loss_pt1(self,
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

        num_components = self.hyperparams['N'] * self.hyperparams['K']
        ### Importance sampled NLL estimate
        z, _ = self.encoder(mode, x, y_e)  # [k_eval, nbs, N*K]
        z = self.latent.sample_p(1, mode, full_dist=True)
        y_dist, _ = self.p_y_xz(ModeKeys.PREDICT, x, x_nr_t, y_r, n_s_t0, z,
                                prediction_horizon, num_samples=1, num_components=num_components)
        # We use unstandardized labels to compute the loss
        log_p_yt_xz = torch.clamp(y_dist.log_prob(labels), max=self.hyperparams['log_p_yt_xz_max'])
        log_p_y_xz = torch.sum(log_p_yt_xz, dim=2)
        log_p_y_xz_mean = torch.mean(log_p_y_xz, dim=0)  # [nbs]
        return x, log_p_y_xz_mean

def train_loss_pt2(log_p_y_xz_mean, kl_term, inf_term):
    log_likelihood = torch.mean(log_p_y_xz_mean)

    ELBO = log_likelihood - kl_term + inf_term
    loss = -ELBO
    return loss

def eval_loss_pt2(log_p_y_xz_mean):
    log_likelihood = torch.mean(log_p_y_xz_mean)
    nll = -log_likelihood
    return nll