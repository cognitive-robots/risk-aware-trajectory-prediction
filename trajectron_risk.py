import torch
import numpy as np
import sys
sys.path.append("Trajectron-plus-plus/trajectron/")
from model.trajectron import Trajectron 
from model.dataset.preprocessing import restore
from mgcvae_risk import MultimodalGenerativeCVAERisk, train_loss_pt2, eval_loss_pt2
from preprocessing_risk import get_timesteps_data
import torch.nn as nn
import wandb

NUM_ENSEMBLE = [0, 1]

def create_stacking_model(env, x_size):
    num_models = len(NUM_ENSEMBLE)
    models = {}
    for node_type in env.NodeType:
        models[node_type] = nn.Sequential(
                                nn.Linear(x_size[node_type]*num_models, num_models).cuda(),
                                nn.ReLU())
    return models

class TrajectronRisk(Trajectron):

    def set_environment(self, env):
        self.env = env

        self.node_models_dict.clear()
        edge_types = env.get_edge_types()

        x_size = {}
        for ens_index in NUM_ENSEMBLE:
            self.node_models_dict[ens_index] = {}
            for node_type in env.NodeType:
                # Only add a Model for NodeTypes we want to predict
                if node_type in self.pred_state.keys():
                    self.node_models_dict[ens_index][node_type] = MultimodalGenerativeCVAERisk(env,
                                                                                node_type,
                                                                                self.model_registrar,
                                                                                self.hyperparams,
                                                                                self.device,
                                                                                edge_types,
                                                                                log_writer=self.log_writer)
                    if node_type not in x_size.keys():
                        x_size[node_type] = self.node_models_dict[ens_index][node_type].x_size
        self.x_size = x_size

    def get_x_size(self):
        return self.x_size

    def bagging(self, losses, encoded_inputs=None, node_type=None, predict=False): # losses is either losses or predictions
        return torch.mean(torch.stack(losses), dim=0)

    def stacking(self, losses, encoded_inputs, node_type, predict=False): # losses is either losses or predictions
        if self.num_models == 1:
            return losses[0]
        model_input = torch.cat(tuple(encoded_inputs), 1).to(self.device)
        model_output = self.agg_models[node_type](model_input) + 0.00001 # to keep from getting nans

        if predict:
            predictions = losses # just indicating that if predict is true, the losses are actually predictions
            sum_preds = torch.zeros(predictions[0].shape).to(self.device)
            for i in range(self.num_models):
                for j in range(model_output.shape[0]): # per example in batch
                    sum_preds[:,j,:,:] += predictions[i][:,j,:,:]*model_output[j,i]
            return sum_preds / model_output.sum() # return normalized weighted average of predictions
            # ind = torch.argmax(model_output) # return predictions of most probably correct model
            # return predictions[ind]

        # do a weighted average of losses, weighted by model_output
        losses_stacked = torch.stack(losses, dim=1)
        weighted_losses = losses_stacked * model_output
        normalized_weighted_average = torch.sum(weighted_losses, 1) / torch.sum(model_output, 1)
        return normalized_weighted_average        
    
    def set_aggregation(self, ensemble_method, agg_models=None):
        num_models = len(NUM_ENSEMBLE)
        self.ensemble_method = ensemble_method
        self.num_models = num_models

        if ensemble_method == 'bag':
            self.aggregation_func = self.bagging

        if ensemble_method == 'stack':
            self.agg_models = agg_models
            self.aggregation_func = self.stacking

    def set_curr_iter(self, curr_iter):
        self.curr_iter = curr_iter
        for ens_str, node_dict in self.node_models_dict.items():
            for node_str, model in node_dict.items():
                model.set_curr_iter(curr_iter)

    def set_annealing_params(self):
        for ens_str, node_dict in self.node_models_dict.items():
            for node_str, model in node_dict.items():
                model.set_annealing_params()

    def step_annealers(self, node_type=None):
        for ens_index in NUM_ENSEMBLE:
            if node_type is None:
                for node_type in self.node_models_dict:
                    self.node_models_dict[ens_index][node_type].step_annealers()
            else:
                self.node_models_dict[ens_index][node_type].step_annealers()

    def train_loss(self, batch, node_type, heatmap_tensor, grid_tensor):
        (first_history_index,
         x_t, y_t, x_st_t, y_st_t,
         neighbors_data_st,
         neighbors_edge_value,
         robot_traj_st_t,
         map,
         #--------------ADDED--------------
         x_unf_t,
         map_name
         #---------------------------------
         ) = batch

        x = x_t.to(self.device)
        y = y_t.to(self.device)
        #--------------ADDED--------------
        x_unf = x_unf_t.to(self.device)
        #---------------------------------
        x_st_t = x_st_t.to(self.device)
        y_st_t = y_st_t.to(self.device)
        if robot_traj_st_t is not None:
            robot_traj_st_t = robot_traj_st_t.to(self.device)
        if type(map) == torch.Tensor:
            map = map.to(self.device)
        
        if self.ensemble_method == 'bag':
            losses = []
            for ens_index in NUM_ENSEMBLE: # for each model in ensemble
                # Run forward pass
                model = self.node_models_dict[ens_index][node_type]
                loss = model.train_loss(inputs=x,
                                        inputs_st=x_st_t,
                                        first_history_indices=first_history_index,
                                        labels=y,
                                        labels_st=y_st_t,
                                        neighbors=restore(neighbors_data_st),
                                        neighbors_edge_value=restore(neighbors_edge_value),
                                        robot=robot_traj_st_t,
                                        map=map,
                                        prediction_horizon=self.ph,
                                        heatmap_tensor=heatmap_tensor,
                                        x_unf=x_unf,
                                        map_name=map_name,
                                        grid_tensor=grid_tensor)
                losses.append(loss)
            ret = self.aggregation_func(losses)
            return ret

        if self.ensemble_method == 'stack':
            losses = []
            encoded_inputs = []
            kls = []
            infs = []
            for ens_index in NUM_ENSEMBLE: # for each model in ensemble
                # Run forward pass
                model = self.node_models_dict[ens_index][node_type]
                encoded_input, per_example_loss, kl_term, inf_term = model.train_loss_pt1(inputs=x,
                                        inputs_st=x_st_t,
                                        first_history_indices=first_history_index,
                                        labels=y,
                                        labels_st=y_st_t,
                                        neighbors=restore(neighbors_data_st),
                                        neighbors_edge_value=restore(neighbors_edge_value),
                                        robot=robot_traj_st_t,
                                        map=map,
                                        prediction_horizon=self.ph,
                                        heatmap_tensor=heatmap_tensor,
                                        x_unf=x_unf,
                                        map_name=map_name,
                                        grid_tensor=grid_tensor)
                wandb.log({"{} train_loss_{}".format(str(node_type), 
                                ens_index): train_loss_pt2(per_example_loss, kl_term, inf_term).item()})
                losses.append(per_example_loss)
                encoded_inputs.append(encoded_input)
                kls.append(kl_term)
                infs.append(inf_term)
            aggregated = self.aggregation_func(losses, encoded_inputs, node_type)
            aggregated_kl = torch.mean(torch.stack(kls)) # mean aggregated kl and inf - 
            aggregated_inf = torch.mean(torch.stack(infs)) # could make it weighted maybe keep learned weighting just for inputs
            return train_loss_pt2(aggregated, aggregated_kl, aggregated_inf)

    def eval_loss(self, batch, node_type):
        (first_history_index,
         x_t, y_t, x_st_t, y_st_t,
         neighbors_data_st,
         neighbors_edge_value,
         robot_traj_st_t,
         map,
         #--------------ADDED--------------
         x_unf_t,
         map_name
         #---------------------------------
         ) = batch

        x = x_t.to(self.device)
        y = y_t.to(self.device)
        x_st_t = x_st_t.to(self.device)
        y_st_t = y_st_t.to(self.device)
        if robot_traj_st_t is not None:
            robot_traj_st_t = robot_traj_st_t.to(self.device)
        if type(map) == torch.Tensor:
            map = map.to(self.device)

        if self.ensemble_method == 'bag':
            nlls = []
            for ens_index in NUM_ENSEMBLE: # for each model in ensemble
                # Run forward pass
                model = self.node_models_dict[ens_index][node_type]
                nll = model.eval_loss(inputs=x,
                                    inputs_st=x_st_t,
                                    first_history_indices=first_history_index,
                                    labels=y,
                                    labels_st=y_st_t,
                                    neighbors=restore(neighbors_data_st),
                                    neighbors_edge_value=restore(neighbors_edge_value),
                                    robot=robot_traj_st_t,
                                    map=map,
                                    prediction_horizon=self.ph)
                nlls.append(nll)
            ret = self.aggregation_func(nlls)
            return ret.cpu().detach().numpy()

        if self.ensemble_method == 'stack':
            nlls = []
            encoded_inputs = []
            for ens_index in NUM_ENSEMBLE: # for each model in ensemble
                # Run forward pass
                model = self.node_models_dict[ens_index][node_type]
                encoded_input, nll = model.eval_loss_pt1(inputs=x,
                                    inputs_st=x_st_t,
                                    first_history_indices=first_history_index,
                                    labels=y,
                                    labels_st=y_st_t,
                                    neighbors=restore(neighbors_data_st),
                                    neighbors_edge_value=restore(neighbors_edge_value),
                                    robot=robot_traj_st_t,
                                    map=map,
                                    prediction_horizon=self.ph)
                wandb.log({"{} eval_loss_{}".format(str(node_type), ens_index): eval_loss_pt2(nll).item()})
                nlls.append(nll)
                encoded_inputs.append(encoded_input)
            aggregated = self.aggregation_func(nlls, encoded_inputs, node_type)
            return eval_loss_pt2(aggregated).cpu().detach().numpy()

    def predict(self,
                scene,
                timesteps,
                ph,
                num_samples=1,
                min_future_timesteps=0,
                min_history_timesteps=1,
                z_mode=False,
                gmm_mode=False,
                full_dist=True,
                all_z_sep=False):

        predictions_dict = {}
        for node_type in self.env.NodeType:
            if node_type not in self.pred_state:
                continue
            all_models_predictions = []
            encoded_inputs = []
            for ens_index in NUM_ENSEMBLE: # for each model in ensemble

                model = self.node_models_dict[ens_index][node_type]

                # Get Input data for node type and given timesteps
                batch = get_timesteps_data(env=self.env, scene=scene, t=timesteps, node_type=node_type, state=self.state,
                                        pred_state=self.pred_state, edge_types=model.edge_types,
                                        min_ht=min_history_timesteps, max_ht=self.max_ht, min_ft=min_future_timesteps,
                                        max_ft=min_future_timesteps, hyperparams=self.hyperparams)
                # There are no nodes of type present for timestep
                if batch is None:
                    # print('BATCH IS NONE')
                    continue
                (first_history_index,
                x_t, y_t, x_st_t, y_st_t,
                neighbors_data_st,
                neighbors_edge_value,
                robot_traj_st_t,
                map,
                #--------------ADDED--------------
                x_unf_t,
                map_name
                #---------------------------------
                ), nodes, timesteps_o = batch

                x = x_t.to(self.device)
                x_st_t = x_st_t.to(self.device)
                if robot_traj_st_t is not None:
                    robot_traj_st_t = robot_traj_st_t.to(self.device)
                if type(map) == torch.Tensor:
                    map = map.to(self.device)

                # Run forward pass
                encoded_input, predictions = model.predict(inputs=x,
                                            inputs_st=x_st_t,
                                            first_history_indices=first_history_index,
                                            neighbors=neighbors_data_st,
                                            neighbors_edge_value=neighbors_edge_value,
                                            robot=robot_traj_st_t,
                                            map=map,
                                            prediction_horizon=ph,
                                            num_samples=num_samples,
                                            z_mode=z_mode,
                                            gmm_mode=gmm_mode,
                                            full_dist=full_dist,
                                            all_z_sep=all_z_sep)
                all_models_predictions.append(predictions)
                encoded_inputs.append(encoded_input)

            if all_models_predictions == []:
                continue
            aggregated_ensemble_predictions = self.aggregation_func(all_models_predictions, 
                                                                    encoded_inputs, node_type, 
                                                                    predict=True)
            predictions_np = aggregated_ensemble_predictions.cpu().detach().numpy()

            # Assign predictions to node
            for i, ts in enumerate(timesteps_o):
                if ts not in predictions_dict.keys():
                    predictions_dict[ts] = dict()
                predictions_dict[ts][nodes[i]] = np.transpose(predictions_np[:, [i]], (1, 0, 2, 3))

        return predictions_dict

    def get_vel(self,
                scene,
                timesteps,
                ph,
                num_samples=1,
                min_future_timesteps=0,
                min_history_timesteps=1,
                z_mode=False,
                gmm_mode=False,
                full_dist=True,
                all_z_sep=False):

        
        for node_type in self.env.NodeType:
            if node_type not in self.pred_state:
                continue

            ens_index = NUM_ENSEMBLE[0]
            model = self.node_models_dict[ens_index][node_type]

            # Get Input data for node type and given timesteps
            batch = get_timesteps_data(env=self.env, scene=scene, t=timesteps, node_type=node_type, state=self.state,
                                       pred_state=self.pred_state, edge_types=model.edge_types,
                                       min_ht=min_history_timesteps, max_ht=self.max_ht, min_ft=min_future_timesteps,
                                       max_ft=min_future_timesteps, hyperparams=self.hyperparams)
            # There are no nodes of type present for timestep
            
            if batch is None:
                continue
            (first_history_index,
             x_t, y_t, x_st_t, y_st_t,
             neighbors_data_st,
             neighbors_edge_value,
             robot_traj_st_t,
             map,
            #--------------ADDED--------------
             x_unf_t,
             map_name
            #---------------------------------
            ), nodes, timesteps_o = batch
            vel_list = []
            vel_array = np.array(vel_list)
            x = x_t.to(self.device)
            #
            for iter in range(len(x)):
                vel_stack = torch.stack((x[iter,:,2], x[iter,:,3]), axis = -1)
                vel_norm = np.linalg.norm(vel_stack.cpu(), axis=-1)
                vel_norm = vel_norm[~np.isnan(vel_norm)]
                vel_array = np.append(vel_array, vel_norm[:])

            average_batch_vel = np.mean(vel_array)
        return average_batch_vel