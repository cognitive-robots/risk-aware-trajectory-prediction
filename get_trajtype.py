import numpy as np
import torch

class TrajectoryType:
    STATIONARY = 0
    STRAIGHT = 1
    STRAIGHT_RIGHT = 2
    STRAIGHT_LEFT = 3
    RIGHT_U_TURN = 4
    RIGHT_TURN = 5
    LEFT_U_TURN = 6
    LEFT_TURN = 7

def classify_track_single(start_point, end_point, start_velocity, end_velocity, start_heading, end_heading):
    # The classification strategy is taken from
    # waymo_open_dataset/metrics/motion_metrics_utils.cc#L28

    # Parameters for classification, taken from WOD
    kMaxSpeedForStationary = 2.0  # (m/s)
    kMaxDisplacementForStationary = 5.0  # (m)
    kMaxLateralDisplacementForStraight = 5.0  # (m)
    kMinLongitudinalDisplacementForUTurn = -5.0  # (m)
    kMaxAbsHeadingDiffForStraight = np.pi / 6.0  # (rad)

    x_delta = end_point[0] - start_point[0]
    y_delta = end_point[1] - start_point[1]

    final_displacement = np.hypot(x_delta, y_delta)
    heading_diff = end_heading - start_heading
    normalized_delta = np.array([x_delta, y_delta])
    rotation_matrix = np.array([[np.cos(-start_heading), -np.sin(-start_heading)],
                                [np.sin(-start_heading), np.cos(-start_heading)]])
    normalized_delta = np.dot(rotation_matrix, normalized_delta)
    start_speed = np.hypot(start_velocity[0], start_velocity[1])
    end_speed = np.hypot(end_velocity[0], end_velocity[1])
    max_speed = max(start_speed, end_speed)
    dx, dy = normalized_delta

    # Check for different trajectory types based on the computed parameters.
    if max_speed < kMaxSpeedForStationary and final_displacement < kMaxDisplacementForStationary:
        return TrajectoryType.STATIONARY
    if np.abs(heading_diff) < kMaxAbsHeadingDiffForStraight:
        if np.abs(normalized_delta[1]) < kMaxLateralDisplacementForStraight:
            return TrajectoryType.STRAIGHT
        return TrajectoryType.STRAIGHT_RIGHT if dy < 0 else TrajectoryType.STRAIGHT_LEFT
    if heading_diff < -kMaxAbsHeadingDiffForStraight and dy < 0:
        return TrajectoryType.RIGHT_U_TURN if normalized_delta[
                                                  0] < kMinLongitudinalDisplacementForUTurn else TrajectoryType.RIGHT_TURN
    if dx < kMinLongitudinalDisplacementForUTurn:
        return TrajectoryType.LEFT_U_TURN
    return TrajectoryType.LEFT_TURN

def classify_track(start_point, end_point, start_velocity, end_velocity, start_heading, end_heading):
    # The classification strategy is taken from
    # waymo_open_dataset/metrics/motion_metrics_utils.cc#L28

    # Parameters for classification, taken from WOD
    kMaxSpeedForStationary = 2.0  # (m/s)
    kMaxDisplacementForStationary = 5.0  # (m)
    kMaxLateralDisplacementForStraight = 5.0  # (m)
    kMinLongitudinalDisplacementForUTurn = -5.0  # (m)
    kMaxAbsHeadingDiffForStraight = np.pi / 6.0  # (rad)

    x_delta = end_point[:,0] - start_point[:,0]
    y_delta = end_point[:,1] - start_point[:,1]

    final_displacement = torch.hypot(x_delta, y_delta) # [batchsize,1]
    heading_diff = end_heading - start_heading # [batchsize, 1]
    normalized_delta_unrot = torch.stack((x_delta, y_delta), dim=-1) #[batchsize, 2]
    rotation_matrix = torch.stack((torch.stack((torch.cos(-start_heading),
                                               torch.sin(-start_heading)),
                                               dim=-1),
                                   torch.stack((-torch.sin(-start_heading),
                                                torch.cos(-start_heading)),
                                               dim=-1)),
                                   dim=-1) # [batchsize,2,2]
    normalized_delta = torch.bmm(normalized_delta_unrot.unsqueeze(1),rotation_matrix).squeeze()
    start_speed = torch.hypot(start_velocity[:,0], start_velocity[:,1])
    end_speed = torch.hypot(end_velocity[:,0], end_velocity[:,1])
    max_speed = torch.max(start_speed, end_speed)
    dx = normalized_delta[:,0]
    dy = normalized_delta[:,1]
    
    ret = torch.zeros(dx.shape) - 1
    # Check for different trajectory types based on the computed parameters.
    ret[torch.logical_and(ret == -1, torch.logical_and(
        max_speed < kMaxSpeedForStationary, 
        final_displacement < kMaxDisplacementForStationary))] = TrajectoryType.STATIONARY
    ret[torch.logical_and(ret == -1, torch.logical_and(
        torch.abs(heading_diff) < kMaxAbsHeadingDiffForStraight, 
        torch.abs(normalized_delta[:,1]) < kMaxLateralDisplacementForStraight))] = TrajectoryType.STRAIGHT
    ret[torch.logical_and(ret == -1, torch.logical_and(
        torch.abs(heading_diff) < kMaxAbsHeadingDiffForStraight, 
        dy < 0))] = TrajectoryType.STRAIGHT_RIGHT
    ret[torch.logical_and(ret == -1, torch.logical_and(
        torch.abs(heading_diff) < kMaxAbsHeadingDiffForStraight, 
        dy >= 0))] = TrajectoryType.STRAIGHT_LEFT
    ret[torch.logical_and(ret == -1, torch.logical_and(
        heading_diff < -kMaxAbsHeadingDiffForStraight, 
        dy < 0))] = TrajectoryType.RIGHT_TURN
    ret[torch.logical_and(ret == TrajectoryType.RIGHT_TURN, 
        normalized_delta[:,0] < kMinLongitudinalDisplacementForUTurn)] = TrajectoryType.RIGHT_U_TURN
    ret[torch.logical_and(ret == -1, 
        dx < kMinLongitudinalDisplacementForUTurn)] = TrajectoryType.LEFT_U_TURN
    ret[ret == -1] = TrajectoryType.LEFT_TURN
    return ret

def get_heading(trajectory):
    # trajectory has shape (Time X (x,y))

    dx_ = np.diff(trajectory[:, 0])
    dy_ = np.diff(trajectory[:, 1])
    heading = np.arctan2(dy_, dx_)

    return heading

def get_trajectory_type(output):
    for data_sample in output:
        # Get last gt position, velocity and heading
        valid_end_point = int(data_sample["center_gt_final_valid_idx"])
        end_point = data_sample["obj_trajs_future_state"][0, valid_end_point, :2]  # (x,y)
        end_velocity = data_sample["obj_trajs_future_state"][0, valid_end_point, 2:]  # (vx, vy)
        # Get last heading, manually approximate it from the series of future position
        end_heading = get_heading(data_sample["obj_trajs_future_state"][0, :valid_end_point + 1, :2])[-1]

        # Get start position, velocity and heading.
        assert data_sample["obj_trajs_mask"][0, -1]  # Assumes that the start point is always valid
        start_point = data_sample["obj_trajs"][0, -1, :2]  # (x,y)
        start_velocity = data_sample["obj_trajs"][0, -1, -4:-2]  # (vx, vy)
        start_heading = 0.  # Initial heading is zero

        # Classify the trajectory
        try:
            trajectory_type = classify_track(start_point, end_point, start_velocity, end_velocity, start_heading,
                                             end_heading)
        except:
            trajectory_type = -1
        data_sample["trajectory_type"] = trajectory_type
    return

# x_t is [batch_size, hist_lenth=9, kinematics=8] 
# where kinematics is [x_position, y_position, x_velocity, y_velocity, x_acc, y_acc, heading_angle, something else]
def get_traj_type(x_t, node_type): 

    start_point = x_t[:,0,:2] # [batchsize, 2] (x,y)
    end_point = x_t[:,-1,:2] # [batchsize, 2] (x,y)

    start_velocity = x_t[:,0,2:4] # [batchsize, 2] (vx, vy)
    end_velocity = x_t[:,-1,2:4] # [batchsize, 2] (vx, vy)

    # if str(node_type) == 'VEHICLE':
    #     start_heading = x_t[:,0,6] # [batchsize, 1] 
    #     end_heading = x_t[:,-1,6] # [batchsize, 1] 
    # else:
    #     heading = get_heading(x_t) # not correct shapes as of now
    # ret = classify_track(start_point, end_point, start_velocity, end_velocity, 
    #                             start_heading, end_heading)
    # import pdb; pdb.set_trace()
    ret = torch.zeros(x_t.shape[0], dtype=torch.int)
    for i in range(len(ret)):
        if str(node_type) == 'VEHICLE':
            ret[i] = classify_track_single(start_point[i], end_point[i], 
                                        start_velocity[i], end_velocity[i], 
                                        x_t[i,0,6], x_t[i,-1,6])
        else:
            heading = get_heading(x_t[i])
            ret[i] = classify_track_single(start_point[i], end_point[i], 
                                        start_velocity[i], end_velocity[i], 
                                        heading[0], heading[-1])


    return ret
