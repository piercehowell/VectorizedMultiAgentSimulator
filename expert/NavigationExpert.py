'''
This is a proof of concept expert.
The goal of this is to act as a template on how to create the real expert once the policy architecture is created
This is not meant to be the final code that we are using
'''

import argparse
import torch
from torch.nn import ModuleDict
import time
import os
import xml.etree.ElementTree as ET
from PIL import Image
from vmas import make_env
import numpy as np
import torch
from tensordict import TensorDict
import pickle

stepsPerTimeInterval = 20 #This still assumes constant velocity... Needs to be large enough so agents actually reach their goal before moving on

def load_map(map_file, device):
    #Load in map
    f = os.path.dirname(__file__)
    map = ET.parse(f'{f}/{map_file}')
    map_root = map.getroot()

    # Extract grid dimensions
    width = int(map_root.find('.//width').text)
    height = int(map_root.find('.//height').text)

    # Initialize a numpy array to store the grid
    grid = torch.zeros((height, width), dtype=torch.int, device=device)

    # Find all row elements and populate the numpy array
    row_elements = map_root.findall('.//row')
    for i, row_element in enumerate(row_elements):
        row_data = row_element.text.split()
        for j, value in enumerate(row_data):
            grid[height - 1 - i, j] = int(value)
    return grid, width, height

def load_episodes(log_dir, device, width, height):
    f = os.path.dirname(__file__)
    episodes = {}
    for file in os.listdir(f'{f}/{log_dir}/'):
        if file[-3:] != 'xml':
            continue

        #episode name must be a number!
        #Assumes the files are named <what ever you want>_<id>_log.xml
        episode_name = int(file[:-4].split('_')[-2])
        
        agent_paths = {}
        log = ET.parse(f'{f}/{log_dir}/{file}')
        log_root = log.getroot()
        agents = log_root.findall(".//agent[@number]")
        for agent in agents:
            number = int(agent.get('number'))
            waypoints = []
            path = agent.find('.//path')
            sections = path.findall('.//section')
            first = True
            for waypoint in sections:
                if first:
                    y = height - 1 - int(waypoint.get('start_i')) #i is row, j is column...
                    x = int(waypoint.get('start_j'))
                    duration = 0.0
                    waypoints.append(torch.tensor([duration, x, y], device=device))
                    first=False
                y = height - 1 - int(waypoint.get("goal_i"))
                x = int(waypoint.get('goal_j'))
                duration = float(waypoint.get('duration')) + waypoints[-1][0]
                waypoints.append(torch.tensor([duration, x, y], device=device))
            agent_paths[number] = waypoints
        episodes[episode_name] = agent_paths.copy()
    return episodes

def select_action(position, waypoint, steps):   
    # Constants
    total_time = .1 * (stepsPerTimeInterval - steps%stepsPerTimeInterval)
    max_linear_velocity = 2.0  # Maximum linear velocity
    max_angular_velocity = torch.pi/3  # Maximum angular velocity

    # Unpack robot position
    x, y, theta = position

    # Unpack desired position
    x_d, y_d = waypoint

    # Calculate the difference in position and angle
    delta_x = x_d - x
    delta_y = y_d - y
    delta_theta = np.arctan2(delta_y, delta_x) - theta

    # Ensure angle difference is within -pi to pi range
    delta_theta = np.arctan2(np.sin(delta_theta), np.cos(delta_theta))

    # Calculate required linear and angular velocities
    v = np.sqrt(delta_x**2 + delta_y**2) / total_time
    w = delta_theta / total_time

    # Cap linear and angular velocities to maximum values
    v = min(v, max_linear_velocity)
    w = min(w, max_angular_velocity)
    w = max(w, -max_angular_velocity)
    return [v, 0, w] #The second parameter is ignored

def get_agent_action(episodes, obs, steps):
    scenario_time = steps / stepsPerTimeInterval
    actions = []
    for i, ob in enumerate(obs):
        env_actions = []
        for env_ob in ob:
            episode = episodes[int(env_ob[0])]
            x, y, theta = env_ob[1:4]
            appended = False
            for waypoint in episode[i]:
                if waypoint[0] > scenario_time:
                    env_actions.append(select_action([x,y,theta],waypoint[1:], steps))
                    appended = True
                    break
            if not appended:
                env_actions.append(select_action([x,y,theta],episode[i][-1][1:], steps))
        actions.append(env_actions)
        
    return actions


def main(args):
    device = args.device  # or cuda or any other torch device

    #Load in the config to get agent sizes and the time limit
    f = os.path.dirname(__file__)
    config = ET.parse(f'{f}/{args.config}')
    config_root = config.getroot()
    agent_size = float(config_root.find('.//agent_size').text)

    #load map. Currently assumes every task takes place on the same map
    grid_map, width, height = load_map(args.map, device)
    
    #Load in all of the pre generated episodes
    episodes = load_episodes(args.log_dir, device, width, height)

    scenario_name = 'navigation2' #Scenario name

    # Scenario specific variables
    n_agents = len(episodes[next(iter(episodes))].keys()) #Assumes all episodes has same number of agents

    num_envs = args.num_envs  # Number of vectorized environments
    n_steps = args.max_steps # Number of steps before returning done

    start_action = (
        [0, 0, 0]
    )  # Simple action to start the program 

    env = make_env(
        scenario=scenario_name,
        num_envs=num_envs,
        device=device,
        dict_spaces=False,
        wrapper=None,
        seed=None,
        # Environment specific variables
        n_agents=n_agents,
        episodes = episodes,
        map=grid_map,
        agent_radius = agent_size/2
    )
    
    frame_list = []  # For creating a gif
    actions = {} 
    obs = []
    episodes_list = []
    for e in range(len(episodes)):
        print('Episode:',e)
        step = 0
        step_list = []
        for s in range(n_steps):
            step += 1
            
            if len(obs) == 0:
                agent_actions = [[start_action] * num_envs]* len(env.agents)
            else:
                agent_actions = get_agent_action(episodes, obs, step)

            # for i, agent in enumerate(env.agents):
            #     action = torch.tensor(
            #         agent_actions[i],
            #         device=device,
            #     )
            #     actions.update({agent.name: action})

            obs, rews, dones, info = env.step(agent_actions)
    
            if args.save_gif and e == 0:
                frame_list.append(
                    Image.fromarray(env.render(mode="rgb_array", agent_index_focus=None))
                )

            step_data = TensorDict({
                'actions': agent_actions.copy(),
                'observations': [o[:, 1:] for o in obs],
                'rewards': rews.copy(),
                'dones': dones.clone(),
            }, batch_size=[])
            step_list.append(step_data)

            if dones.all():
                break
        
        if args.save_gif and e == 0:
            gif_name = scenario_name + ".gif"
            # Produce a gif
            frame_list[0].save(
                gif_name,
                save_all=True,
                append_images=frame_list[1:],
                duration=3,
                loop=0,
            )

        episodes_list.append(step_list)
        if e < len(episodes)-1:
            env.reset()

    file_path = f"{args.log_dir}/trajectories.pkl"
    with open(file_path, 'wb') as file:
        pickle.dump(episodes_list, file)

def create_parser():
    parser = argparse.ArgumentParser(description='Use the expert to solve the navigation2 scenario')
    parser.add_argument('--map', '-m', default='grid_map.xml', type=str, help='The map to generate tasks for. Only supports grid type maps')
    parser.add_argument('--config', '-c', default='config.xml', help='The config file for the task')
    parser.add_argument('--log_dir', '-l', default='logs', help='local path to logs directory')
    parser.add_argument('--device', '-d', default = 'cpu', help='device to run on')
    parser.add_argument('--num_envs', type=int, default = 1, help='number of environments to run at once')
    parser.add_argument('--save_gif', action="store_true", help='whether or not to save the first episode as a gif')
    parser.add_argument('--max_steps', type=int, default = 500, help='Max number of steps to run for')
    return parser

if __name__ == "__main__":
    args = create_parser().parse_args()
    main(args)