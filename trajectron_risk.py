import torch
import numpy as np
from scipy.cluster.vq import vq, whiten, kmeans2
import sys
sys.path.append("Trajectron-plus-plus/trajectron/")
from model.trajectron import Trajectron 
from model.dataset.preprocessing import restore
from mgcvae_encoder import MultimodalGenerativeCVAEEncoder
from mgcvae_decoder import MultimodalGenerativeCVAEDecoder, train_loss_pt2, eval_loss_pt2
from preprocessing_risk import get_timesteps_data
import torch.nn as nn
import wandb
from skimage.util import random_noise
from sklearn.cluster import MiniBatchKMeans

STACKING_CHOOSE_ONE = False # for stack only
INCR_ETA = False

# if we're sweeping over these, these are ignored
stacking_model_eta = 0.1 # for stack or stackboost

# only for deep_clustering, the epoch is defined in train_risk for per_epoch deep_clustering
CLUSTERSTACK_EPOCH = -1 # use -1 if want clustering for all epochs

vicreg_eta = 0.1

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
                                nn.Linear(hidden_layer_size, output_layer_size).to(device),
                                nn.BatchNorm1d(hidden_layer_size).to(device),
                                nn.ReLU()
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

def vicreg_loss(x, y, sim_coeff=25, std_coeff=25, cov_coeff=1):
        batch_size = x.shape[0]
        num_features = x.shape[1]

        repr_loss = F.mse_loss(x, y)

        # x = torch.cat(FullGatherLayer.apply(x), dim=0)
        # y = torch.cat(FullGatherLayer.apply(y), dim=0)
        x = x - x.mean(dim=0)
        y = y - y.mean(dim=0)

        std_x = torch.sqrt(x.var(dim=0) + 0.0001)
        std_y = torch.sqrt(y.var(dim=0) + 0.0001)
        std_loss = torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2

        cov_x = (x.T @ x) / (batch_size - 1)
        cov_y = (y.T @ y) / (batch_size - 1)
        cov_loss = off_diagonal(cov_x).pow_(2).sum().div(
            num_features
        ) + off_diagonal(cov_y).pow_(2).sum().div(num_features)

        loss = (
            sim_coeff * repr_loss
            + std_coeff * std_loss
            + cov_coeff * cov_loss
        )
        return loss

def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

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
        for node_type in env.NodeType:
            # Only add a Model for NodeTypes we want to predict
            if node_type in self.pred_state.keys():
                self.clusters[node_type] = None
                self.node_models_dict[node_type] = {}
                # 1 encoder
                encoder = MultimodalGenerativeCVAEEncoder(env,
                                                    node_type,
                                                    '_encoder',
                                                    self.model_registrar,
                                                    self.hyperparams,
                                                    self.device,
                                                    edge_types,
                                                    log_writer=self.log_writer)                
                self.node_models_dict[node_type]['encoder'] = encoder
                x_size[node_type] = encoder.x_size
                z_dim[node_type] = encoder.latent.z_dim
                zx[node_type] = encoder.latent.z_dim + encoder.x_size

                # num_ensemble number of decoders
                for ens_index in num_ensemble:
                    decoder = MultimodalGenerativeCVAEDecoder(env,
                                                        node_type,
                                                        ens_index,
                                                        self.model_registrar,
                                                        self.hyperparams,
                                                        self.device,
                                                        edge_types,
                                                        log_writer=self.log_writer)
                    decoder.prev_gmm_params = None # only needed for boosting
                    self.node_models_dict[node_type][ens_index] = decoder

        self.x_size = x_size
        self.z_dim = z_dim
        self.zx_dim = zx

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

        loss = nn.CrossEntropyLoss()

        self.stacking_model_loss = loss(model_output, target.to(self.device)) * self.stack_eta
        return model_inference

    def set_aggregation(self, ensemble_method, agg_models=None, percentage=None, eta=None):
        num_models = len(self.num_ensemble)
        self.ensemble_method = ensemble_method
        self.num_models = num_models
        if not eta:
            eta = stacking_model_eta

        if 'clusterstack' in ensemble_method:
            self.agg_models = agg_models
            self.aggregation_func = self.clusterstacking
            self.stack_eta = eta
            self.kmeans = {}

    def set_curr_iter(self, curr_iter):
        self.curr_iter = curr_iter
        for node, ens_index_dict in self.node_models_dict.items():
            for ens_index, model in ens_index_dict.items():
                model.set_curr_iter(curr_iter)

    def set_annealing_params(self):
        for node, ens_index_dict in self.node_models_dict.items():
            for ens_index, model in ens_index_dict.items():
                model.set_annealing_params()

    def step_annealers(self, node_type=None):
        for ens_index in self.num_ensemble:
            if node_type is None:
                for node_type in self.node_models_dict:
                    self.node_models_dict[node_type][ens_index].step_annealers()
            else:
                self.node_models_dict[node_type][ens_index].step_annealers()

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

        if 'clusterstack' in self.ensemble_method:
            if epoch < CLUSTERSTACK_EPOCH:
                encoder = self.node_models_dict[node_type]['encoder'] # only the first model's encoder is used
                z, kl, input_embedding, dists = encoder.train_loss_encode(
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
                # loss for every model, average model losses
                losses = []
                for ens_index in self.num_ensemble: # for each model in ensemble
                    # Run forward pass
                    model = self.node_models_dict[node_type][ens_index]
                    per_example_likelihood, kl_term, inf_term = model.train_loss_decode_pt1(
                                                                z, kl, input_embedding, dists,
                                                                labels=y, 
                                                                prediction_horizon=self.ph)
                    loss = train_loss_pt2(per_example_likelihood, kl_term, inf_term)
                    wandb.log({"{} train_loss_{}".format(str(node_type), ens_index): loss.item()})                                                                
                    losses.append(loss)

                ret = torch.mean(torch.stack(losses), dim=0)
                return ret

            # approach is: 1 encoder, many decoders

            # Encoder
            encoder = self.node_models_dict[node_type]['encoder'] # only the first model's encoder is used
            z, kl, input_embedding, dists = encoder.train_loss_encode(
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

            # vicreg_loss_term = vicreg_loss(zx_features, zx_features_prime) * vicreg_eta


            if CLUSTERSTACK_EPOCH == -1: #do clustering every epoch (normal)
                if self.ensemble_method == 'clusterstackperepoch':
                    stacking_label = self.kmeans[node_type].predict(whitened)
                else:
                    if self.clusters[node_type] is None:
                        centroid, stacking_label = kmeans2(whitened, k=self.num_models, iter=10)
                    else: 
                        centroid, stacking_label = kmeans2(whitened, k=self.clusters[node_type], minit='matrix', iter=10)
                    self.clusters[node_type] = centroid

                target = torch.tensor(stacking_label)

            else: #do clustering only for clusterstack_epoch
                if epoch == CLUSTERSTACK_EPOCH:
                    if self.clusters[node_type] is None:
                        centroid, stacking_label = kmeans2(whitened, k=self.num_models, iter=100)
                    else: 
                        centroid, stacking_label = kmeans2(whitened, k=self.clusters[node_type], minit='matrix', iter=100)
                    self.clusters[node_type] = centroid
                    target = torch.tensor(stacking_label)
                elif epoch > CLUSTERSTACK_EPOCH:
                    # just calculate labels based on self.clusters
                    code, dist = vq(whitened, self.clusters[node_type])
                    target = torch.tensor(code)

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
                decoder = self.node_models_dict[node_type][ens_index]            
                per_example_likelihood, kl_term, inf_term = decoder.train_loss_decode_pt1(
                                                            z, kl, input_embedding, dists,
                                                            labels=y, 
                                                            prediction_horizon=self.ph)
                this_models_losses = per_example_likelihood[mask == ens_index]
                per_example_losses.append(this_models_losses)
                loss = train_loss_pt2(this_models_losses, kl_term, inf_term)
                wandb.log({"{} train_loss_{}".format(str(node_type), ens_index): loss.item()})

            all_losses = torch.cat(per_example_losses)
            total_loss = train_loss_pt2(all_losses, kl_term, inf_term)
            return total_loss + self.stacking_model_loss #+ vicreg_loss_term

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
            
        if 'clusterstack' in self.ensemble_method:
            # Encoder
            encoder = self.node_models_dict[node_type]['encoder'] 
            z, input_embedding, dists = encoder.eval_loss_encode(
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
                model = self.node_models_dict[node_type][ens_index]
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
            # Encoder
            encoder = self.node_models_dict[node_type]['encoder'] 
            encoder_output = encoder.predict_encode(inputs=x,
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
            # Get Input data for node type and given timesteps
            batch = get_timesteps_data(env=self.env, scene=scene, t=timesteps, node_type=node_type, state=self.state,
                                    pred_state=self.pred_state, edge_types=encoder.edge_types,
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

            for ens_index in num_ensemble_subset: # for each model in ensemble
                # Run forward pass
                decoder = self.node_models_dict[node_type][ens_index]
                predictions = decoder.predict_decode(inputs=x,
                                            inputs_st=x_st_t,
                                            first_history_indices=first_history_index,
                                            neighbors=neighbors_data_st,
                                            neighbors_edge_value=neighbors_edge_value,
                                            robot=robot_traj_st_t,
                                            map=map,
                                            prediction_horizon=ph,
                                            num_samples=num_samples,
                                            encoder_output=encoder_output,
                                            z_mode=z_mode,
                                            gmm_mode=gmm_mode,
                                            full_dist=full_dist,
                                            all_z_sep=all_z_sep)
                all_models_predictions.append(predictions)
                encoded_inputs.append(x)

            if all_models_predictions == []:
                continue

            if 'clusterstack' in self.ensemble_method:
                x = encoder_output[0]
                z = encoder_output[-1]
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

    # cluster functions
    def cluster_init(self, node_type, batch_size):
        if self.clusters[node_type] is None:
            minibatch_kmeans = MiniBatchKMeans(n_clusters=self.num_models,
                                    max_iter=10,
                                    # random_state=0, # seed - use for debugging if needed
                                    batch_size=batch_size,
                                    n_init=1)
        else:
            minibatch_kmeans = MiniBatchKMeans(n_clusters=self.num_models,
                                    init=self.clusters[node_type],
                                    max_iter=10,
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
            # map = torch.tensor(random_noise(map.cpu()), dtype=torch.float) #Gaussian distributed additive noise (randgaussmapnoise)
            map = map.to(self.device)
        # Encoder
        encoder = self.node_models_dict[node_type]['encoder'] 
        z, kl, input_embedding, dists = encoder.train_loss_encode(
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
            model = self.node_models_dict[node_type][ens_index]

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

    
    def viz_clusters(self,
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

                model = self.node_models_dict[node_type][ens_index]
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
                if ('clusterstack' in self.ensemble_method) and ens_index == 0:
                    clusterstack_encoder_output = encoder_output

                if (self.ensemble_method == 'boost') or (self.ensemble_method == 'gradboost'):
                    gmm_params_prod = model.gmm_params # model's p_y_xz to use (prev) gmm_params_prod * new predicted gmm

            if all_models_predictions == []:
                continue

            features = encoded_inputs
            if ('clusterstack' in self.ensemble_method):
                x = clusterstack_encoder_output[0]
                z = clusterstack_encoder_output[-1]

                # if features are zx not x:
                z_example_per_mode = torch.reshape(z, (-1, self.z_dim[node_type]))
                features = torch.cat([z_example_per_mode, x.repeat(z.shape[0], 1)], dim=1) 
            
                self.agg_models[node_type].eval()
                model_output = self.agg_models[node_type](features) + 0.00001 # to keep from getting nans, [256]
                model_output = model_output.softmax(dim=1) # need it to predict a class (ie a model)
                model_inference = torch.argmax(model_output, dim=1) # index of most probably correct model, [256]
                
                # if features are zx not x:
                predictions = all_models_predictions
                inds_per_sample = model_inference.reshape(-1, predictions[0].shape[1])
                ret = torch.zeros(predictions[0].shape).to(self.device)
                for i in range(inds_per_sample.shape[0]):
                    for j in range(inds_per_sample.shape[1]): # per example in batch
                        ind = inds_per_sample[i, j]
                        ret[i,j,:,:] = predictions[ind][i,j,:,:]
                aggregated_ensemble_predictions = ret
            
            predictions_np = aggregated_ensemble_predictions.cpu().detach().numpy()

            # Assign predictions to node
            for i, ts in enumerate(timesteps_o):
                if ts not in predictions_dict.keys():
                    predictions_dict[ts] = dict()
                predictions_dict[ts][nodes[i]] = np.transpose(predictions_np[:, [i]], (1, 0, 2, 3))
                import pdb; pdb.set_trace()

        return predictions_dict