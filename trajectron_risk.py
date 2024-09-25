import torch
import numpy as np
from scipy.cluster.vq import vq, whiten, kmeans2
from sklearn.cluster import MiniBatchKMeans
import sys
sys.path.append("Trajectron-plus-plus/trajectron/")
from model.trajectron import Trajectron 
from model.dataset.preprocessing import restore
from mgcvae_risk import MultimodalGenerativeCVAERisk, train_loss_pt2, eval_loss_pt2
from preprocessing_risk import get_timesteps_data
import torch.nn as nn
import wandb
from skimage.util import random_noise


STACKING_CHOOSE_ONE = False # for stack only
INCR_ETA = False

# if we're sweeping over these, these are ignored
STACKBOOST_PERCENTAGE = 0.5 # for stackboost
stacking_model_eta = 0.1 # for stack or stackboost

def create_stacking_model(env, model_registrar, input_dims, device, input_multiplier, num_models):
    models = {}
    for node_type in env.NodeType:
        input_layer_size = input_dims[node_type]*input_multiplier
        output_layer_size = num_models
        hidden_layer_size =  input_layer_size
        model_if_absent = nn.Sequential(
                                nn.Linear(input_layer_size, hidden_layer_size).to(device),
                                nn.BatchNorm1d(hidden_layer_size).to(device),
                                nn.ReLU(),
                                nn.Linear(hidden_layer_size, output_layer_size).to(device)
                                )
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
        z_dim = {}
        zx = {}
        self.clusters = {}
        for ens_index in num_ensemble:
            self.node_models_dict[ens_index] = {}
            for node_type in env.NodeType:
                # Only add a Model for NodeTypes we want to predict
                if node_type in self.pred_state.keys():
                    self.clusters[node_type] = None
                    model_instance = MultimodalGenerativeCVAERisk(env,
                                                        node_type,
                                                        ens_index,
                                                        self.model_registrar,
                                                        self.hyperparams,
                                                        self.device,
                                                        edge_types,
                                                        log_writer=self.log_writer)
                    model_instance.prev_gmm_params = None # only needed for boosting
                    self.node_models_dict[ens_index][node_type] = model_instance
                    if node_type not in x_size.keys():
                        x_size[node_type] = model_instance.x_size
                        z_dim[node_type] = model_instance.latent.z_dim
                        zx[node_type] = model_instance.latent.z_dim + model_instance.x_size
        self.x_size = x_size
        self.z_dim = z_dim
        self.zx_dim = zx

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

    def clusterstacking(self, target, encoded_inputs, node_type, eval=False, predict=False): # losses is either losses or predictions
        # device = self.agg_models[self.env.NodeType[0]][0].weight.device
        model_input = encoded_inputs.to(self.device)

        if predict: # 20 samples answers coming from same mode
            self.agg_models[node_type].eval()
            model_output = self.agg_models[node_type](model_input) + 0.00001 # to keep from getting nans, [256]
            model_output = model_output.softmax(dim=1) # need it to predict a class (ie a model)
            model_inference = torch.argmax(model_output, dim=1) # index of most probably correct model, [256]
            # if features are zx not x:
            predictions = target
            inds_per_mode = model_inference.reshape(-1, target[1].shape[1]).transpose(0,1)
            inds = torch.mode(inds_per_mode).values
            ret = torch.zeros(predictions[0].shape).to(self.device)
            for i in range(inds.shape[0]):
                ind = inds[i]
                ret[:,i,:,:] = predictions[ind][:,i,:,:]
            return ret 

        model_output = self.agg_models[node_type](model_input) + 0.00001 # to keep from getting nans, [256]
        model_output = model_output.softmax(dim=1) # need it to predict a class (ie a model)
        model_inference = torch.argmax(model_output, dim=1) # index of most probably correct model, [256]
        if eval:
            # if features are zx not x:
            inds_per_mode = model_inference.reshape(-1, target[0].shape[0]).transpose(0,1)
            inds = torch.mode(inds_per_mode).values
            losses_stacked = torch.stack(target)
            return losses_stacked[inds, range(len(inds))] # return predictions of most probably correct model
        # if predict: # 20 samples answers coming from different modes
        #     # if features are zx not x:
        #     predictions = target
        #     inds_per_sample = model_inference.reshape(-1, target[0].shape[1])
        #     ret = torch.zeros(predictions[0].shape).to(self.device)
        #     for i in range(inds_per_sample.shape[0]):
        #         for j in range(inds_per_sample.shape[1]): # per example in batch
        #             ind = inds_per_sample[i, j]
        #             ret[i,j,:,:] = predictions[ind][i,j,:,:]
        #     return ret 
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

        if ensemble_method == 'clusterstack':
            self.agg_models = agg_models
            self.aggregation_func = self.clusterstacking
            self.stack_eta = eta
            self.kmeans = {}

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
            # map = torch.tensor(random_noise(map.cpu()), dtype=torch.float) #Gaussian distributed additive noise (randgaussmapnoise)
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

        if self.ensemble_method == 'clusterstack':
            if epoch < self.clusterstack_epoch:
                # loss for every model, average model losses
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
                ret = self.bagging(losses)
                return ret

            # approach is: 1 encoder, many decoders

            # Encoder
            model = self.node_models_dict[0][node_type] # only the first model's encoder is used
            z, kl, input_embedding, dists = model.train_loss_encode(
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
            x = input_embedding[0]

            # cluster the zx into num_models clusters
            z_example_per_mode = torch.reshape(z, (-1, self.z_dim[node_type]))
            zx_features = torch.cat([z_example_per_mode, x.repeat(self.z_dim[node_type], 1)], dim=1)       
            whitened = whiten(zx_features.cpu().detach().numpy())

            # # if doing per_batch clustering:
            # if self.clusters[node_type] is None:
            #     centroid, stacking_label = kmeans2(whitened, k=self.num_models, iter=10)
            # else: 
            #     centroid, stacking_label = kmeans2(whitened, k=self.clusters[node_type], minit='matrix', iter=10)
            # self.clusters[node_type] = centroid

            # if doing per_epoch clustering:
            stacking_label = self.kmeans[node_type].predict(whitened)
            target = torch.tensor(stacking_label)

            # # if features are x, not zx:
            # target_per_mode = target.reshape(self.z_dim[node_type], -1).transpose(0,1)
            # target = torch.mode(target_per_mode).values

            # run stacking_classifier on zx (or x)
            mask = self.aggregation_func(target.to(torch.int64), zx_features, node_type) 

            # if features are zx not x:
            mask_per_mode = mask.reshape(self.z_dim[node_type], -1).transpose(0,1)
            mask = torch.mode(mask_per_mode).values

            # Decoder
            per_example_losses = [] # per-model likelihoods
            for ens_index in self.num_ensemble: # for each model in ensemble
                # Run every model's decoder
                model = self.node_models_dict[ens_index][node_type]            
                per_example_likelihood, kl_term, inf_term = model.train_loss_decode_pt1(
                                            z, kl, input_embedding, dists,
                                            labels=y, 
                                            prediction_horizon=self.ph)
                this_models_losses = per_example_likelihood[mask == ens_index]
                per_example_losses.append(this_models_losses)
                loss = train_loss_pt2(this_models_losses, kl_term, inf_term)
                wandb.log({"{} train_loss_{}".format(str(node_type), ens_index): loss.item()})

            all_losses = torch.cat(per_example_losses)
            total_loss = train_loss_pt2(all_losses, kl_term, inf_term)
            return total_loss + self.stacking_model_loss

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
            
        if self.ensemble_method == 'clusterstack':
            # Encoder
            model = self.node_models_dict[0][node_type] # only the first model's encoder is used
            z, input_embedding, dists = model.eval_loss_encode(
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
            x = input_embedding[0]
            # if features are zx not x:
            z_example_per_mode = torch.reshape(z, (-1, self.z_dim[node_type]))
            zx_features = torch.cat([z_example_per_mode, x.repeat(z.shape[0], 1)], dim=1)       

            nlls = []
            for ens_index in self.num_ensemble: # for each model in ensemble
                # Run forward pass
                model = self.node_models_dict[ens_index][node_type]
                encoded_input, nll = model.eval_loss_decode_pt1(
                                            z, input_embedding, dists,
                                            labels=y, 
                                            prediction_horizon=self.ph)
                wandb.log({"{} eval_loss_{}".format(str(node_type), ens_index): eval_loss_pt2(nll).item()})
                nlls.append(nll)
            aggregated = self.aggregation_func(nlls, zx_features, node_type, eval=True)
            return eval_loss_pt2(aggregated).cpu().detach().numpy()

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
            clusterstack_encoder_output = None
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
                encoder_output, predictions = model.predict(inputs=x,
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
                x = encoder_output[0]
                z = encoder_output[-1]
                all_models_predictions.append(predictions)
                encoded_inputs.append(x)
                if (self.ensemble_method == 'clusterstack') and ens_index == 0:
                    clusterstack_encoder_output = encoder_output

                if (self.ensemble_method == 'boost') or (self.ensemble_method == 'gradboost'):
                    gmm_params_prod = model.gmm_params # model's p_y_xz to use (prev) gmm_params_prod * new predicted gmm

            if all_models_predictions == []:
                continue

            features = encoded_inputs
            if (self.ensemble_method == 'clusterstack'):
                x = clusterstack_encoder_output[0]
                z = clusterstack_encoder_output[-1]
                # if features are zx not x:
                z_example_per_mode = torch.reshape(z, (-1, self.z_dim[node_type]))
                features = torch.cat([z_example_per_mode, x.repeat(z.shape[0], 1)], dim=1) 

            aggregated_ensemble_predictions = self.aggregation_func(all_models_predictions, 
                                                                    features, node_type, 
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

    def cluster_init(self, node_type, batch_size):
        if self.clusters[node_type] is None:
            minibatch_kmeans = MiniBatchKMeans(n_clusters=self.num_models,
                                    max_iter=100,
                                    # random_state=0, # seed - use for debugging if needed
                                    batch_size=batch_size,
                                    n_init=1)
        else:
            minibatch_kmeans = MiniBatchKMeans(n_clusters=self.num_models,
                                    init=self.clusters[node_type],
                                    max_iter=100,
                                    # random_state=0, # seed - use for debugging if needed
                                    batch_size=batch_size,
                                    n_init=1)
        self.kmeans[node_type] = minibatch_kmeans

    # for per_epoch clustering:
    def clustering_step(self, zx_features, node_type):
        whitened = whiten(zx_features.cpu().detach().numpy()) + 0.000000001 # for stable PSD matrix
        self.kmeans[node_type] = self.kmeans[node_type].partial_fit(whitened)

    def set_clusters(self, node_type):
        self.clusters[node_type] = self.kmeans[node_type].cluster_centers_

    # for per_batch clustering:
    # def clustering_step(self, zx_features, node_type):
    #     whitened = whiten(zx_features.cpu().detach().numpy()) + 0.000000001 # for stable PSD matrix
    #     if self.clusters[node_type] is None:
    #         centroid, _ = kmeans2(whitened, k=self.num_models, iter=10)
    #     else: 
    #         centroid, _ = kmeans2(whitened, k=self.clusters[node_type], minit='matrix', iter=10)
    #     self.clusters[node_type] = centroid

    def get_encoded_features(self, batch, node_type, heatmap_tensor, grid_tensor, epoch, gradboost_index=None):
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
            map = torch.tensor(random_noise(map.cpu()), dtype=torch.float) #Gaussian distributed additive noise (randgaussmapnoise)
            map = map.to(self.device)

        # Encoder
        model = self.node_models_dict[0][node_type] # only the first model's encoder is used
        z, kl, input_embedding, dists = model.train_loss_encode(
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
        x = input_embedding[0]

        # cluster the zx into num_models clusters
        z_example_per_mode = torch.reshape(z, (-1, self.z_dim[node_type]))
        zx_features = torch.cat([z_example_per_mode, x.repeat(self.z_dim[node_type], 1)], dim=1)       
        return zx_features