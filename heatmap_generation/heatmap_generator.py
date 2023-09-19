import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

import argparse
from nuscenes.nuscenes import NuScenes
from nuscenes.map_expansion.map_api import NuScenesMap
import numpy as np
from numpy.linalg import norm
import pandas as pd
import matplotlib.pyplot as plt
import tqdm

import seaborn as sns; sns.set()
from PIL import Image

version = 'v1.0-trainval'
maps = ['boston-seaport', 'singapore-onenorth', 'singapore-queenstown', 'singapore-hollandvillage']
visualize = False
xgrid_no = 100

def visualize_heatmap(df_map_loc_full_dropped_vru, hist, map_choice, nusc):
    df_map_loc_full_dropped_vru_copy = df_map_loc_full_dropped_vru

    location_list = df_map_loc_full_dropped_vru_copy['grid_location'].tolist()
    grid_loc_count_list = []
    hist_list = hist.tolist()

    for i in range(len(location_list)):
        value_ = location_list[i]
        grid_loc_count_list.append(hist[int(value_)-1])

    df_map_loc_full_dropped_vru_copy['grid_loc_count'] = grid_loc_count_list

    color_fg = (167, 174, 186)
    color_bg = (255, 255, 255)

    # Get logs by location
    log_tokens = [l['token'] for l in nusc.log if l['location'] == map_choice]
    assert len(log_tokens) > 0, 'Error: This split has 0 scenes for location %s!' % map_choice

    # Filter scenes
    scene_tokens_location = [e['token'] for e in nusc.scene if e['log_token'] in log_tokens]
    if len(scene_tokens_location) == 0:
        print('Warning: Found 0 valid scenes for location %s!' % map_choice)
    map_poses = []
    map_mask = None

    #print(scene_tokens_location)
    for scene_token in scene_tokens_location:

        # Get records from the database.
        scene_record = nusc.get('scene', scene_token)
        log_record = nusc.get('log', scene_record['log_token'])
        map_record = nusc.get('map', log_record['map_token'])
        map_mask = map_record['mask']

        # For each sample in the scene, store the ego pose.
        sample_tokens = nusc.field2token('sample', 'scene_token', scene_token)
        for sample_token in sample_tokens:
            sample_record = nusc.get('sample', sample_token)

            # Poses are associated with the sample_data. Here we use the lidar sample_data.
            sample_data_record = nusc.get('sample_data', sample_record['data']['LIDAR_TOP'])
            pose_record = nusc.get('ego_pose', sample_data_record['ego_pose_token'])

            # Calculate the pose on the map and append
            map_poses.append(np.concatenate(
                map_mask.to_pixel_coords(pose_record['translation'][0], pose_record['translation'][1])))

    # Compute number of close ego poses.
    print('Creating plot...')

    if len(np.array(map_mask.mask()).shape) == 3 and np.array(map_mask.mask()).shape[2] == 3:
        # RGB Colour maps.
        mask = map_mask.mask()
    else:
        # Monochrome maps.
        # Set the colors for the mask.
        mask = Image.fromarray(map_mask.mask())
        mask = np.array(mask)

        maskr = color_fg[0] * np.ones(np.shape(mask), dtype=np.uint8)
        maskr[mask == 0] = color_bg[0]
        maskg = color_fg[1] * np.ones(np.shape(mask), dtype=np.uint8)
        maskg[mask == 0] = color_bg[1]
        maskb = color_fg[2] * np.ones(np.shape(mask), dtype=np.uint8)
        maskb[mask == 0] = color_bg[2]
        mask = np.concatenate((np.expand_dims(maskr, axis=2),
                                np.expand_dims(maskg, axis=2),
                                np.expand_dims(maskb, axis=2)), axis=2)

    _, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(mask)
    #print(ax)

    title = 'Heatmap for {}'.format(map_choice)
    ax.set_title(title, color='k')
    #plt.gca().invert_yaxis()
    y_values = (ax.get_ylim()[0] - df_map_loc_full_dropped_vru_copy['location_y']*10)
    sc = ax.scatter(df_map_loc_full_dropped_vru_copy['location_x']*10, y_values, s=10, c=df_map_loc_full_dropped_vru_copy['grid_loc_count'])
    #print(ax.get_ylim()[0])
    #print(ax.get_ylim()[1])
    color_bar = plt.colorbar(sc, fraction=0.025, pad=0.04)
    plt.rcParams['figure.facecolor'] = 'black'
    color_bar_ticklabels = plt.getp(color_bar.ax.axes, 'yticklabels')
    plt.setp(color_bar_ticklabels, color='k')
    plt.rcParams['figure.facecolor'] = 'white'  # Reset for future plots.
    plt.savefig("heatmap"+map_choice+"_"+version+".png")

def load_map(map_choice, dataroot):
    nusc_map = NuScenesMap(dataroot=dataroot, map_name=map_choice)

    # map of whole area is drivable_area[0] for all except hollandvillage, which is drivable_area[2]
    n = 0
    if map_choice == 'singapore-hollandvillage':
        n = 2
    
    sample_drivable_area = nusc_map.drivable_area[n]
    if visualize:
        fig, ax = nusc_map.render_record('drivable_area', sample_drivable_area['token'], other_layers=[])
        plt.show()
    
    #Gather limits of driveable area generated above.
    #Denoting the 100 by 100 grid numbered from left to right, then vertically
    #i.e.
    #1,2,3,...,100,
    #101,102,...,200,
    #201,... etc.
    ax_loc = ax[1]
    y_min = (ax_loc.get_ylim()[0])
    y_max = (ax_loc.get_ylim()[1])
    x_min = (ax_loc.get_xlim()[0])
    x_max = (ax_loc.get_xlim()[1])
    x_grid_size = (x_max - x_min)/grid_no
    y_grid_size = (y_max - y_min)/grid_no

    return (x_min,x_max,y_min,y_max,x_grid_size,y_grid_size)

def create_heatmap(map_choice, dataroot):
    (x_min,x_max,y_min,y_max,x_grid_size,y_grid_size) = load_map(map_choice, dataroot)
    nusc = NuScenes(version=version, dataroot=dataroot, verbose=True)

    # Generate empty dataframe
    store = {
        'scene_token':[],
        'sample_token':[],
        'ann0_token' :[],
        'ann_n':[],
        'location_x':[],
        'location_z':[],
        'location_y':[],
        'grid_location':[],
        'distance':[]
                }

    df = pd.DataFrame(store)
    print('Empty dataframe: ', df)

    #Finding the distance & grid location only between Moving vehicles and VRUs (not inanimate objects this time!);
    #One scene returned 23,000 points.
    df_vru_scene_dict = {}

    for scene in nusc.scene:
    # Loop over ALL scenes

        print(nusc.scene.index(scene))
        sample_tokens = []
        # Empty list
        df_sample_dict = {}
        # Empty dict

        log_token = scene['log_token']
        if nusc.get('log',log_token)['location'] == map_choice:
        #ONLY select scenes that are at 'singapore-onenorth'
            for i in range(scene['nbr_samples']):
            # Loop over number of samples per scene
                if i == 0:
                # If start of loop, take the given "first sample token".
                    sample_token = scene['first_sample_token']
                else:
                # If not at start of loop, take "next" sample token.
                    sample_token = nusc.get('sample', sample_token)['next']
                # Append the token to the list
                sample_tokens.append(sample_token)

            #We have all of 1 scene's sample_tokens listed.
            #Start looping through samples (timesteps).
            for sample_no in range(len(sample_tokens)):
                current_sample = nusc.get('sample', sample_tokens[sample_no])
                ann_list = current_sample['anns']

                #Now loop through annotations at each timestep.
                #Working out the euclid dist betweeen all annotations.
                #No point in comparing 2 non-moving items.

                #Initialise "VRU" list.
                VRU = [None]*len(ann_list)

                #Determine if the annotation is moving
                for x in range(len(ann_list)):
                    att=nusc.get('sample_annotation', ann_list[x])['attribute_tokens']

                    #If the object CANNOT MOVE (len = 0) - i.e. cone, bollard
                    if len(att) == 0:
                        VRU[x] = 0

                    #elif vehicle
                    elif att[0] == nusc.attribute[0]['token'] or att[0] == nusc.attribute[1]['token'] or att[0] == nusc.attribute[1]['token']:
                        VRU[x] = 1

                    #else VRU
                    else:
                        VRU[x] = 2


                #Create a dictionary to store each annotation's dataframes.
                df_ann_dict = {}

                #Remove annotation 0 and compare with all others, and iterate.
                # Compares 0 w/ 1,2,3... then 1 w/ 2,3,4... 2 w/ 3,4,5....
                # No overlapping comparisons.
                while len(ann_list)>1:
                    ann0 = ann_list[0]
                    ann_list = ann_list[1:]

                    df_ann0_dict = {}

                    VRU0 = VRU[0]
                    VRU = VRU[1:]

                    ann0_loc = np.array(nusc.get('sample_annotation', ann0)['translation'])

                    for n in range(len(ann_list)):
                        #if you are NOT exactly Vehicle and VRU, skip.
                        if not VRU0 + VRU[n] == 3:
                            continue

                        else:
                            ann_n_loc = np.array(nusc.get('sample_annotation', ann_list[n])['translation'])
                            translation_difference = ann0_loc - ann_n_loc
                            norm_difference = norm(translation_difference)
                            x_val = (ann0_loc[0]-x_min)//x_grid_size
                            y_val = (ann0_loc[1]-y_min)//y_grid_size
                            grid_loc = (grid_no*(100-y_val)) + x_val
                            info_n = {
                            'scene_token':[scene['token']],
                            'sample_token':[sample_tokens[sample_no]],
                            'ann0_token' :[ann0],
                            'ann_n':[ann_list[n]],
                            'location_x':[ann0_loc[0]],
                            'location_y':[ann0_loc[1]],
                            'location_z':[ann0_loc[2]],
                            'grid_location':[grid_loc],
                            'distance':[norm_difference]
                            }
                            df_ann0_n = pd.DataFrame(info_n)
                            df_ann0_dict[ann_list[n]]=df_ann0_n

                    if len(df_ann0_dict)>0:
                        df_ann_n = pd.concat(df_ann0_dict.values())
                        df_ann_dict[ann0] = df_ann_n

                if len(df_ann_dict)>0:
                    df_sample_n = pd.concat(df_ann_dict.values())
                    df_sample_dict[sample_tokens[sample_no]] = df_sample_n

        if len(df_sample_dict)>0:
            df_scene_n = pd.concat(df_sample_dict.values())
            df_vru_scene_dict[scene['token']] = df_scene_n

    if len(df_vru_scene_dict)>0:
        df_map_vru = pd.concat(df_vru_scene_dict.values())

    # Save DF to CSV.
    file_name_vru = "df_map_VRU_"+map_choice+"_"+version+".csv"
    df_map_vru.to_csv(file_name_vru, encoding='utf-8', index=False)

    # Sorting Riskiest locations
    df_map_vru.sort_values(by=['sample_token','distance'],inplace=True,ascending = [True, True])
    df_map_loc_full_dropped_vru = df_map_vru.drop_duplicates("sample_token")
    df_map_loc_vru = df_map_loc_full_dropped_vru.drop(labels=['scene_token','ann0_token','sample_token','ann_n','location_x','location_z','location_y','distance'],axis=1)

    hist, edges = np.histogram(df_map_loc_vru['grid_location'], bins=(grid_no*grid_no), range=(1, (grid_no*grid_no+1)))

    # Save this map's hist to CSV
    np.savetxt("hist"+map_choice+"_"+version+".csv", hist, delimiter=",")

    if visualize:
        visualize_heatmap(df_map_loc_full_dropped_vru, hist, map_choice, nusc)
    return hist

def create_normalized_csv(dataroot):
    df_hist = {}
    for map_choice in maps:
        if os.path.isfile('hist{}_v1.0-trainval.csv'.format(map_choice)):
            df_hist[map_choice] = pd.read_csv('hist{}_v1.0-trainval.csv'.format(map_choice), header=None)
        else:
            df_hist[map_choice] = create_heatmap(map_choice, dataroot)
    
    df_hist_all_4 = pd.concat((df_hist[map_choice] for map_choice in maps), axis=1)
    df_hist_all_4.columns = maps

    df_hist_all = df_hist_all_4.copy()
    df_hist_all['average'] = df_hist_all.mean(axis=1)

    move_to_front = df_hist_all.pop('average')
    df_hist_all.insert(0, 'average', move_to_front)

    one_zero_normalized_df_hist_all=(df_hist_all-df_hist_all.min())/(df_hist_all.max()-df_hist_all.min())
    one_zero_normalized_df_hist_all.to_csv('one_zero_normalized_df_hist_all.csv', encoding='utf-8', index=False)

    nine_zero_normalized_df_hist_all=one_zero_normalized_df_hist_all*9
    nine_zero_normalized_df_hist_all.to_csv('nine_zero_normalized_df_hist_all.csv', encoding='utf-8', index=False)

    two_one_normalized_df_hist_all = one_zero_normalized_df_hist_all + 1
    two_one_normalized_df_hist_all.to_csv('two_one_normalized_df_hist_all.csv', encoding='utf-8', index=False)

    ten_one_normalized_df_hist_all = nine_zero_normalized_df_hist_all + 1
    ten_one_normalized_df_hist_all.to_csv('ten_one_normalized_df_hist_all.csv', encoding='utf-8', index=False)

def create_map_info_csv(dataroot):
    grid_info_store = {}
    for map_choice in maps:
        grid_info_store[map_choice] = []
    grid_info_df = pd.DataFrame(grid_info_store)

    for map_choice in maps:
        grid_info_df[map_choice] = load_map(map_choice, dataroot)
    
    grid_info_df.to_csv('grid_info_all.csv', encoding='utf-8', index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default="./Trajectron-plus-plus/experiments/nuScenes/v1.0")
    args = parser.parse_args()
    create_normalized_csv(args.data)
    create_map_info_csv(args.data)