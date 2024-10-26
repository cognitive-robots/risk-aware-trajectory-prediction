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

class MultimodalGenerativeCVAEDecoder(MultimodalGenerativeCVAE):
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

    def train_loss_decode_pt1(self, z, kl, input_embedding, dists, labels, prediction_horizon):
        mode = ModeKeys.TRAIN        
        (latent, dynamic) = dists
        self.latent = latent
        self.dynamic = dynamic

        (x, x_nr_t, y_e, y_r, y, n_s_t0) = input_embedding
        
        log_p_y_xz = self.decoder(mode, x, x_nr_t, y, y_r, n_s_t0, z,
                                  labels,  # Loss is calculated on unstandardized label
                                  prediction_horizon,
                                  self.hyperparams['k'])


        log_p_y_xz_mean = torch.mean(log_p_y_xz, dim=0)  # [nbs]
        log_likelihood = torch.mean(log_p_y_xz_mean)

        mutual_inf_q = mutual_inf_mc(self.latent.q_dist)
        mutual_inf_p = mutual_inf_mc(self.latent.p_dist)
            
        kl_term = self.kl_weight * kl
        inf_term = 1. * mutual_inf_p
        return log_p_y_xz_mean, kl_term, inf_term
    
    def eval_loss_decode_pt1(self, z, input_embedding, dists, labels, prediction_horizon):
        (latent, dynamic) = dists
        self.latent = latent
        self.dynamic = dynamic
        mode = ModeKeys.EVAL
        (x, x_nr_t, y_e, y_r, y, n_s_t0) = input_embedding
        num_components = self.hyperparams['N'] * self.hyperparams['K']

        z = self.latent.sample_p(1, mode, full_dist=True)
        y_dist, _ = self.p_y_xz(ModeKeys.PREDICT, x, x_nr_t, y_r, n_s_t0, z,
                                prediction_horizon, num_samples=1, num_components=num_components)
        # We use unstandardized labels to compute the loss
        log_p_yt_xz = torch.clamp(y_dist.log_prob(labels), max=self.hyperparams['log_p_yt_xz_max'])
        log_p_y_xz = torch.sum(log_p_yt_xz, dim=2)
        log_p_y_xz_mean = torch.mean(log_p_y_xz, dim=0)  # [nbs]
        return x, log_p_y_xz_mean

    def predict_decode(self,
                inputs,
                inputs_st,
                first_history_indices,
                neighbors,
                neighbors_edge_value,
                robot,
                map,
                prediction_horizon,
                num_samples,
                encoder_output,
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
        (x, x_nr_t, y_r, n_s_t0, num_samples, num_components, z) = encoder_output


        _, our_sampled_future = self.p_y_xz(mode, x, x_nr_t, y_r, n_s_t0, z,
                                            prediction_horizon,
                                            num_samples,
                                            num_components,
                                            gmm_mode)

        return our_sampled_future

def train_loss_pt2(log_p_y_xz_mean, kl_term, inf_term):
    log_likelihood = torch.mean(log_p_y_xz_mean)

    ELBO = log_likelihood - kl_term + inf_term
    loss = -ELBO
    return loss

def eval_loss_pt2(log_p_y_xz_mean):
    log_likelihood = torch.mean(log_p_y_xz_mean)
    nll = -log_likelihood
    return nll