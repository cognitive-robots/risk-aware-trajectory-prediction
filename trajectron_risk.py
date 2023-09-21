import torch
import numpy as np
import sys
sys.path.append("Trajectron-plus-plus/trajectron/")
from model.trajectron import Trajectron 
from model.dataset.preprocessing import restore
from mgcvae_risk import MultimodalGenerativeCVAERisk
from preprocessing_risk import get_timesteps_data


class TrajectronRisk(Trajectron):

    def set_environment(self, env):
        self.env = env

        self.node_models_dict.clear()
        edge_types = env.get_edge_types()

        for node_type in env.NodeType:
            # Only add a Model for NodeTypes we want to predict
            if node_type in self.pred_state.keys():
                self.node_models_dict[node_type] = MultimodalGenerativeCVAERisk(env,
                                                                            node_type,
                                                                            self.model_registrar,
                                                                            self.hyperparams,
                                                                            self.device,
                                                                            edge_types,
                                                                            log_writer=self.log_writer)

    def train_loss(self, batch, node_type, heatmap_tensor_ped, heatmap_tensor_veh, grid_tensor, loc_risk=False, no_stat=False):
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

        # Run forward pass
        model = self.node_models_dict[node_type]
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
                                # ---------------- ADDED --------------
                                heatmap_tensor_ped=heatmap_tensor_ped,
                                heatmap_tensor_veh=heatmap_tensor_veh,
                                x_unf=x_unf,
                                map_name=map_name,
                                grid_tensor=grid_tensor,
                                loc_risk=loc_risk,
                                no_stat=no_stat
                                # -------------------------------------

                                )
        return loss

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

        # Run forward pass
        model = self.node_models_dict[node_type]
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

        return nll.cpu().detach().numpy()

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

            model = self.node_models_dict[node_type]

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

            x = x_t.to(self.device)
            x_st_t = x_st_t.to(self.device)
            if robot_traj_st_t is not None:
                robot_traj_st_t = robot_traj_st_t.to(self.device)
            if type(map) == torch.Tensor:
                map = map.to(self.device)

            # Run forward pass
            predictions = model.predict(inputs=x,
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

            predictions_np = predictions.cpu().detach().numpy()

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

            model = self.node_models_dict[node_type]
            #import pdb; pdb.set_trace()
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
            #import pdb; pdb.set_trace()
            average_batch_vel = np.mean(vel_array)
        return average_batch_vel