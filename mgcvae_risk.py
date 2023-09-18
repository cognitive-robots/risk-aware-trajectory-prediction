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


class MultimodalGenerativeCVAERisk(MultimodalGenerativeCVAE):

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
                   grid_tensor,
                   loc_risk,
                   no_stat
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
        # 256x1 torch tensor
        # #---------ADDED--------
        sum_of_weights = 0
        for iter in range(len(map_name)):
            if map_name[iter] != 0: #map successful
                risk_weight = 1
                # Stack the velocities of all entries in the batch.
                vel_stack = torch.stack((inputs[iter,:,2], inputs[iter,:,3]), axis = -1)
                vel_norm = np.linalg.norm(vel_stack.cpu(), axis=-1)
                vel_norm = vel_norm[~np.isnan(vel_norm)]

                if no_stat:
                    ### UNWEIGHT STATIONARY
                    if np.sum(vel_norm) == 0:
                        if str(self.node_type) == 'VEHICLE':
                            risk_weight = 0
                if loc_risk:
                    ###RISKY LOCATION
                    
                    ##describing what each variable is:
                    #grid_tensor=x_min,x_max,y_min,y_max,x_grid_size,y_grid_size: 
                    #for maps boston-seaport, singapore-onenorth, singapore-queenstown, singapore-hollandvillage
                    
                    ##Work out grid loc from unf_x
                    map_number = map_name[iter].item()
                    map_integer = int(map_number)
                    x_val = (x_unf[iter][-1][0]-grid_tensor[0][map_integer-1])//grid_tensor[4][map_integer-1]
                    y_val = (x_unf[iter][-1][1]-grid_tensor[2][map_integer-1])//grid_tensor[5][map_integer-1]

                    # Bear in mind, this assumes a 100x100 grid 

                    if y_val == 0:
                        y_val = 1
                    if x_val == 100:
                        x_val = 99
                    grid_loc = (100*(100-y_val)) + x_val
                    grid_loc_int = int(grid_loc.item())
                    if grid_loc_int > 10000:
                        log_p_y_xz_mean[iter] = log_p_y_xz_mean[iter] 
                        sum_of_weights = sum_of_weights + 1
                    #Multiply Loss term
                    else:
                        log_p_y_xz_mean[iter] = log_p_y_xz_mean[iter] * heatmap_tensor[grid_loc_int][map_integer]
                        sum_of_weights = sum_of_weights + heatmap_tensor[grid_loc_int][map_integer]

            else:
                log_p_y_xz_mean[iter] = log_p_y_xz_mean[iter] 
                sum_of_weights = sum_of_weights + 1
        # -----------------------
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