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

STACKING_CHOOSE_ONE = False # for stack only
STACKBOOST_PERCENTAGE = 0.5 # for stackboost
stacking_model_eta = 0.1 # for stack or stackboost
INCR_ETA = False

# if we're sweeping over these, these are ignored
# STACKBOOST_PERCENTAGE = 0.5 # for stackboost
# stacking_model_eta = 0.1 # for stack or stackboost

def create_stacking_model(env, model_registrar, x_size, device, num_ensemble):
    num_models = len(num_ensemble)
    models = {}
    for node_type in env.NodeType:
        input_layer_size = x_size[node_type]*num_models
        output_layer_size = num_models
        # hidden_layer_size =  int((input_layer_size + output_layer_size) / 2)
        hidden_layer_size1 = int(2/3 * input_layer_size + output_layer_size)
        # hidden_layer_size2 = int(2/3 * hidden_layer_size1 + output_layer_size)
        # hidden_layer_size3 = int(2/3 * hidden_layer_size2 + output_layer_size)
        # hidden_layer_size4 = int(2/3 * hidden_layer_size3 + output_layer_size)
        model_if_absent = nn.Sequential(
                                nn.Linear(input_layer_size, hidden_layer_size1).to(device),
                                nn.ReLU(),
                                nn.Linear(hidden_layer_size1, output_layer_size).to(device),
                                # nn.ReLU(),
                                # nn.Linear(hidden_layer_size, output_layer_size).to(device),
                                # nn.ReLU(),
                                # nn.Linear(hidden_layer_size, hidden_layer_size).to(device),
                                # nn.ReLU(),
                                # nn.Linear(hidden_layer_size, output_layer_size).to(device),
                                nn.ReLU())
        models[node_type] = model_registrar.get_model(str(node_type) + '/stacking_model', model_if_absent)
    return models

def mask_neighbors(neighbors, cond): #cond is an if condition like (mask == ens_index)
    neighbors_masked = {}
    neighbor_inds = torch.where(cond)[0].tolist()
    restored_neighbors = restore(neighbors)
    for key in restored_neighbors.keys():
        neighbors_masked[key] = [restored_neighbors[key][i] for i in neighbor_inds]
    return neighbors_masked

class TrajectronRisk(Trajectron):

    def set_environment(self, env, num_ensemble):
        self.env = env
        self.num_ensemble = num_ensemble

        self.node_models_dict.clear()
        edge_types = env.get_edge_types()

        x_size = {}
        for ens_index in num_ensemble:
            self.node_models_dict[ens_index] = {}
            for node_type in env.NodeType:
                # Only add a Model for NodeTypes we want to predict
                if node_type in self.pred_state.keys():
                    self.node_models_dict[ens_index][node_type] = MultimodalGenerativeCVAERisk(env,
                                                                                node_type,
                                                                                ens_index,
                                                                                self.model_registrar,
                                                                                self.hyperparams,
                                                                                self.device,
                                                                                edge_types,
                                                                                log_writer=self.log_writer)
                    self.node_models_dict[ens_index][node_type].prev_gmm_params = None # only needed for boosting
                    if node_type not in x_size.keys():
                        x_size[node_type] = self.node_models_dict[ens_index][node_type].x_size
        self.x_size = x_size

    def get_x_size(self):
        return self.x_size

    def bagging(self, losses, encoded_inputs=None, node_type=None, predict=False): # losses is either losses or predictions
        return torch.mean(torch.stack(losses), dim=0)
    
    def boosting(self, losses, encoded_inputs=None, node_type=None, predict=False): # losses is either losses or predictions
        if predict:
            predictions = losses # if predict is true, the losses are actually predictions
            return predictions[-1] # last model has used gmms of all prev models
        
        return torch.mean(torch.stack(losses), dim=0)

    def stacking(self, losses, encoded_inputs, node_type, predict=False): # losses is either losses or predictions
        if self.num_models == 1:
            return losses[0]
        model_input = torch.cat(tuple(encoded_inputs), 1).to(self.device)
        model_output = self.agg_models[node_type].to(self.device)(model_input) + 0.00001 # to keep from getting nans

        if STACKING_CHOOSE_ONE:
            losses_stacked = torch.stack(losses)
            if not predict: # compute loss for stacking model
                losses_transposed = torch.transpose(losses_stacked, 0, 1)
                loss = nn.CrossEntropyLoss()
                softmax = model_output.softmax(dim=1)
                target = torch.argmax(losses_transposed, dim=1) # this is max because it's log_likelihood, not nll yet
                self.stacking_model_loss = loss(softmax, target) * self.stack_eta

            inds = torch.argmax(model_output, dim=1) # return predictions of most probably correct model
            return losses_stacked[inds, range(len(inds))]
        
        if predict:
            predictions = losses # just indicating that if predict is true, the losses are actually predictions
            sum_preds = torch.zeros(predictions[0].shape).to(self.device)
            for i in range(self.num_models):
                for j in range(model_output.shape[0]): # per example in batch
                    sum_preds[:,j,:,:] += predictions[i][:,j,:,:]*model_output[j,i]
            return sum_preds / model_output.sum() # return normalized weighted average of predictions

        # do a weighted average of losses, weighted by model_output
        losses_stacked = torch.stack(losses, dim=1)
        weighted_losses = losses_stacked * model_output
        normalized_weighted_average = torch.sum(weighted_losses, 1) / torch.sum(model_output, 1)
        return normalized_weighted_average  

    def stackboosting(self, target, encoded_inputs, node_type, predict=False): # losses is either losses or predictions
        # device = self.agg_models[self.env.NodeType[0]][0].weight.device
        model_input = torch.cat(tuple(encoded_inputs), 1).to(self.device)
        model_output = self.agg_models[node_type](model_input) + 0.00001 # to keep from getting nans, [256]
        model_output = model_output.softmax(dim=1) # need it to predict a class (ie a model)
        model_inference = torch.argmax(model_output, dim=1) # index of most probably correct model, [256]
        if predict:
            predictions = target # just indicating that if predict is true, the target is actually predictions
            if self.num_models == 1:
                return predictions[0]
            agg_preds = torch.zeros(predictions[0].shape).to(self.device)
            for i in range(model_output.shape[0]): # per example in batch
                agg_preds[:,i,:,:] = predictions[model_inference[i]][:,i,:,:] # ith example gets predictions from chosen ens at i
            return agg_preds
        loss = nn.CrossEntropyLoss()
        self.stacking_model_loss = loss(model_output, target.to(self.device)) * self.stack_eta
        return model_inference

    def set_aggregation(self, ensemble_method, agg_models=None, percentage=None, eta=None):
        num_models = len(self.num_ensemble)
        self.ensemble_method = ensemble_method
        self.num_models = num_models
        if not eta:
            eta = stacking_model_eta
        if not percentage:
            percentage = STACKBOOST_PERCENTAGE

        if ensemble_method == 'bag':
            self.aggregation_func = self.bagging

        if (self.ensemble_method == 'boost') or (self.ensemble_method == 'gradboost'):
            self.aggregation_func = self.boosting

        if ensemble_method == 'stack':
            self.agg_models = agg_models
            self.aggregation_func = self.stacking
            self.stack_eta = eta

        if ensemble_method == 'stackboost':
            self.agg_models = agg_models
            self.aggregation_func = self.stackboosting
            self.stackboost_percentage = percentage
            self.stack_eta = eta

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
        for ens_index in self.num_ensemble:
            if node_type is None:
                for node_type in self.node_models_dict:
                    self.node_models_dict[ens_index][node_type].step_annealers()
            else:
                self.node_models_dict[ens_index][node_type].step_annealers()

    def train_loss(self, batch, node_type, heatmap_tensor, grid_tensor, epoch, gradboost_index=None):
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

        if self.ensemble_method == 'gradboost':
            ens_index = gradboost_index
            model = self.node_models_dict[ens_index][node_type]

            if ens_index > 0:
                prev_model = self.node_models_dict[ens_index-1][node_type]
                nll = prev_model.eval_loss(inputs=x,
                                    inputs_st=x_st_t,
                                    first_history_indices=first_history_index,
                                    labels=y,
                                    labels_st=y_st_t,
                                    neighbors=restore(neighbors_data_st),
                                    neighbors_edge_value=restore(neighbors_edge_value),
                                    robot=robot_traj_st_t,
                                    map=map,
                                    prediction_horizon=self.ph)
                model.prev_gmm_params = prev_model.gmm_params # sets the prev_gmm_params FOR THIS BATCH

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
            return loss
        
        if self.ensemble_method == 'bag':
            losses = []
            for ens_index in self.num_ensemble: # for each model in ensemble
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

        if self.ensemble_method == 'boost':
            losses = []
            gmm_params_prod = None
            for ens_index in self.num_ensemble: # for each model in ensemble
                # Run forward pass
                model = self.node_models_dict[ens_index][node_type]
                model.prev_gmm_params = gmm_params_prod # tells model's p_y_xz function to use product of prev_gmm and new predicted gmm
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
                gmm_params_prod = model.gmm_params
                
            ret = self.aggregation_func(losses)
            if torch.any(~torch.isfinite(ret)):
                import pdb; pdb.set_trace()
            return ret

        if self.ensemble_method == 'stack':
            losses = []
            encoded_inputs = []
            kls = []
            infs = []
            for ens_index in self.num_ensemble: # for each model in ensemble
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
            if STACKING_CHOOSE_ONE:
                if INCR_ETA:
                    return train_loss_pt2(aggregated, aggregated_kl, aggregated_inf) + self.stacking_model_loss * epoch # incr stack loss every epoch
                return train_loss_pt2(aggregated, aggregated_kl, aggregated_inf) + self.stacking_model_loss # add stack choosing model loss
            return train_loss_pt2(aggregated, aggregated_kl, aggregated_inf)

        if self.ensemble_method == 'stackboost':
            batch_size = first_history_index.shape[0]
            last_model_index = self.num_models - 1
            ind_k = batch_size
            mask = torch.zeros_like(first_history_index)
            losses = []
            encoded_inputs = []
            for ens_index in self.num_ensemble: # for each model in ensemble
                # Run forward pass
                model = self.node_models_dict[ens_index][node_type]
                encoded_input, per_example_likelihood, kl_term, inf_term = model.train_loss_pt1(
                                        inputs=x,
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
                encoded_inputs.append(encoded_input)
                per_example_likelihood[mask != ens_index] = float('inf') # ignore examples belonging to prev models (since we're getting min k)
                ind_k = int(self.stackboost_percentage * ind_k)
                inds = torch.topk(per_example_likelihood, ind_k, largest=False, sorted=True) # get min k (worst=> smallest likelihood)
                if ens_index != last_model_index: # if no more models, leave leftovers to last model
                    mask[inds.indices] += 1 # leave the ones it got wrong to the next model
                this_models_losses = per_example_likelihood[mask == ens_index]
                loss = train_loss_pt2(this_models_losses, kl_term, inf_term)
                wandb.log({"{} train_loss_{}".format(str(node_type), ens_index): loss.item()})
                losses.append(loss)

            _ = self.aggregation_func(mask, encoded_inputs, node_type) 
            if INCR_ETA:
                return torch.mean(torch.stack(losses)) + self.stacking_model_loss * epoch # incr stack loss weight every epoch
            return torch.mean(torch.stack(losses)) + self.stacking_model_loss

    def eval_loss(self, batch, node_type, gradboost_index=None):
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
            for ens_index in self.num_ensemble: # for each model in ensemble
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

        if self.ensemble_method == 'boost':
            nll = None
            gmm_params_prod = None
            for ens_index in self.num_ensemble: # for each model in ensemble
                # Run forward pass
                model = self.node_models_dict[ens_index][node_type]
                model.prev_gmm_params = gmm_params_prod # tells model's p_y_xz function to use product of prev_gmm and new predicted gmm
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
                gmm_params_prod = model.gmm_params

            return nll.cpu().detach().numpy() # use just the nll of the latest model (which uses product of all predicted gmms)

        if self.ensemble_method == 'gradboost':
            ens_index = gradboost_index
            model = self.node_models_dict[ens_index][node_type]

            if ens_index > 0:
                prev_model = self.node_models_dict[ens_index-1][node_type]
                nll = prev_model.eval_loss(inputs=x,
                                    inputs_st=x_st_t,
                                    first_history_indices=first_history_index,
                                    labels=y,
                                    labels_st=y_st_t,
                                    neighbors=restore(neighbors_data_st),
                                    neighbors_edge_value=restore(neighbors_edge_value),
                                    robot=robot_traj_st_t,
                                    map=map,
                                    prediction_horizon=self.ph)
                model.prev_gmm_params = prev_model.gmm_params # sets the prev_gmm_params FOR THIS BATCH

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

            return nll.cpu().detach().numpy() # use just the nll of the latest model (which uses product of all predicted gmms)

        if self.ensemble_method == 'stack':
            nlls = []
            encoded_inputs = []
            for ens_index in self.num_ensemble: # for each model in ensemble
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
            
        if self.ensemble_method == 'stackboost':
            batch_size = first_history_index.shape[0]
            last_model_index = self.num_models - 1
            ind_k = batch_size
            mask = torch.zeros_like(first_history_index)
            nlls = []
            encoded_inputs = []
            for ens_index in self.num_ensemble: # for each model in ensemble
                # Run forward pass
                model = self.node_models_dict[ens_index][node_type]
                encoded_input, nll = model.eval_loss_pt1(
                                    inputs=x,
                                    inputs_st=x_st_t,
                                    first_history_indices=first_history_index,
                                    labels=y,
                                    labels_st=y_st_t,
                                    neighbors=restore(neighbors_data_st),
                                    neighbors_edge_value=restore(neighbors_edge_value),
                                    robot=robot_traj_st_t,
                                    map=map,
                                    prediction_horizon=self.ph)
                encoded_inputs.append(encoded_input)
                nlls.append(nll)
                wandb.log({"{} eval_loss_{}".format(str(node_type), ens_index): eval_loss_pt2(nll).item()})

            ens_model = self.aggregation_func(mask, encoded_inputs, node_type)
            agg_preds = [nlls[ens_model[i]][i] for i in range(ens_model.shape[0])]
            ret = eval_loss_pt2(torch.tensor(agg_preds))
            return ret.cpu().detach().numpy()

    def predict(self,
                scene,
                timesteps,
                ph,
                last_model_index = None, 
                num_samples=1,
                min_future_timesteps=0,
                min_history_timesteps=1,
                z_mode=False,
                gmm_mode=False,
                full_dist=True,
                all_z_sep=False):
        
        up_to_model = None
        if last_model_index:
            up_to_model = last_model_index + 1
        num_ensemble_subset = self.num_ensemble[:up_to_model] #should be all the models up to and including last_model_index, if None, all models
        predictions_dict = {}
        for node_type in self.env.NodeType:
            if node_type not in self.pred_state:
                continue
            all_models_predictions = []
            encoded_inputs = []
            gmm_params_prod = None
            for ens_index in num_ensemble_subset: # for each model in ensemble

                model = self.node_models_dict[ens_index][node_type]
                model.prev_gmm_params = gmm_params_prod # None for all except boost

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
                if (self.ensemble_method == 'boost') or (self.ensemble_method == 'gradboost'):
                    gmm_params_prod = model.gmm_params # model's p_y_xz to use (prev) gmm_params_prod * new predicted gmm

            if all_models_predictions == []:
                continue
            aggregated_ensemble_predictions = self.aggregation_func(all_models_predictions, 
                                                                    encoded_inputs, node_type, 
                                                                    predict=True)
            # aggregated_ensemble_predictions = all_models_predictions[-1] # when combining smartly, pick last model (has all combined info) and always do gmm_params_prod
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

            ens_index = self.num_ensemble[0]
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
