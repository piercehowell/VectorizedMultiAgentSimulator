'''
Modified navigation.py for the het-mamp project.

Three intended modes:
1. swap - agents will swap with eachother's positions.
2. alcove - agent has to give way for another agent.
3. navigation - agent has to navigate to goal with obstacles and other agents.
'''

import numpy as np
import os
import torch
import typing
import xml.etree.ElementTree as ET

from torch import Tensor
from typing import Dict, Callable, List
from vmas import render_interactively
from vmas.simulator.core import Action, Agent, Box, Landmark, World, Sphere, Entity
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.sensors import Lidar
from vmas.simulator.utils import Color, ScenarioUtils
from vmas.simulator.controllers.velocity_controller import VelocityController

if typing.TYPE_CHECKING:
    from vmas.simulator.rendering import Geom


class Scenario(BaseScenario):
    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        self.device = device
        self.plot_grid = False
        self.n_agents = kwargs.get("n_agents", 2)
        self.collisions = kwargs.get("collisions", True)

        self.observe_all_goals = kwargs.get("observe_all_goals", False)
        self.agents_with_same_goal = kwargs.get("agents_with_same_goal", 1)
        self.split_goals = kwargs.get("split_goals", False)

        self.capability_aware = kwargs.get("capability_aware", False)
        self.agent_radius = kwargs.get("agent_radius", 0.25)
        self.obstacle_dim = kwargs.get("obstacle_dim", 1)
        self.lidar_range = kwargs.get("lidar_range", self.agent_radius*2)
        self.comms_range = kwargs.get("comms_range", 0)

        self.shared_rew = kwargs.get("shared_rew", True)
        self.pos_shaping_factor = kwargs.get("pos_shaping_factor", 1)
        self.final_reward = kwargs.get("final_reward", 0.01)

        self.agent_agent_collision_penalty = kwargs.get("agent_agent_collision_penalty", -1)
        self.agent_obstacle_collision_penalty = kwargs.get("agent_obstacle_collision_penalty", -1)

        self.map_name = kwargs.get("map", "swap")
        self.map, self.x_bounds, self.y_bounds, self.start_poses, self.goal_poses = self.parse_map(self.map_name)
        
        self.min_distance_between_entities = self.obstacle_dim + self.agent_radius + 0.05
        self.min_collision_distance = 0.005

        controller_params = [2, 6, 0.002]

        # Make world
        world = World(batch_dim, device, substeps=2)

        known_colors = [
            (0.22, 0.49, 0.72),
            (1.00, 0.50, 0),
            (0.30, 0.69, 0.29),
            (0.97, 0.51, 0.75),
            (0.60, 0.31, 0.64),
            (0.89, 0.10, 0.11),
            (0.87, 0.87, 0),
        ]
        colors = torch.randn(
            (max(self.n_agents - len(known_colors), 0), 3), device=device
        )
        entity_filter_agents: Callable[[Entity], bool] = lambda e: isinstance(e, Agent) or isinstance(e, Landmark) and e.collide

        # if running in alcove mode, ensure only 2 agents
        if self.map_name == "alcove":
            assert self.n_agents == 2, "Alcove env requires 2 agents"

        # Add agents
        for i in range(self.n_agents):
            color = (
                known_colors[i]
                if i < len(known_colors)
                else colors[i - len(known_colors)]
            )

            # Constraint: all agents have same action range and multiplier
            agent = Agent(
                name=f"agent {i}",
                collide=self.collisions,
                color=color,
                shape=Sphere(radius=self.agent_radius),
                render_action=True,
                sensors=[
                    Lidar(
                        world,
                        n_rays=12,
                        max_range=self.lidar_range,
                        entity_filter=entity_filter_agents,
                    ),
                ]
                if self.collisions
                else None,
            )
            agent.controller = VelocityController(
                agent, world, controller_params, "standard"
            )
            agent.pos_rew = torch.zeros(batch_dim, device=device)
            agent.agent_collision_rew = agent.pos_rew.clone()
            world.add_agent(agent)

            # Add goals
            goal = Landmark(
                name=f"goal {i}",
                collide=False,
                color=color,
            )
            world.add_landmark(goal)
            agent.goal = goal

        # Add in the map obstacles
        indices = torch.nonzero(self.map == 1)
        for coord in indices:
            obstacle = Landmark(
                name = f"Obstacle at {coord}",
                collide = True,
                shape = Box(self.obstacle_dim, self.obstacle_dim),
                collision_filter = lambda e: isinstance(e, Agent)
            )
            world.add_landmark(obstacle)   

        self.pos_rew = torch.zeros(batch_dim, device=device)
        self.final_rew = self.pos_rew.clone()

        return world 
    
    def parse_map(self, map_name: str):
        # Load in map xml
        f = os.path.dirname(__file__)
        map = ET.parse(f'{f}/../../maps/{map_name}.xml')
        map_root = map.getroot()

        # Extract grid dimensions
        width = int(map_root.find('.//width').text)
        height = int(map_root.find('.//height').text)

        # Initialize a numpy array to store the grid
        grid = torch.zeros((height, width), dtype=torch.int, device=self.device)

        # Parse agent start and goal positions
        start_poses = []
        goal_poses = []
        for agent in map_root.findall('.//agent'):
            start_x = float(agent.get('start_i'))
            start_y = float(agent.get('start_j'))
            goal_x = float(agent.get('goal_i'))
            goal_y = float(agent.get('goal_j'))
            start_poses.append([start_x, start_y])
            goal_poses.append([goal_x, goal_y])

        # Find all row elements and populate the numpy array
        row_elements = map_root.findall('.//row')
        for i, row_element in enumerate(row_elements):
            row_data = row_element.text.split()
            for j, value in enumerate(row_data):
                grid[height - 1 - i, j] = int(value)
        return grid, (0, width), (0, height), start_poses, goal_poses
    
    def _reset_capability_at(self, env_index: int = None):
        """
        Reset agent capabilities.
        When a None index is passed to env_index, the world should make a vectorized (batch) reset of capabilities
        """
        # TODO: Enable loading from specific teams

        # capabilities [max velocity, agent radius]
        for agent in self.world.agents:
            agent_max_v = np.random.uniform(0.10, 0.50)
            agent_radius = green_agent_radius = np.random.uniform(0.10, 0.25)

            # set capability
            agent.set_capability(
                torch.tensor(
                    [agent_max_v, agent_radius],
                    dtype=torch.float32,
                    device=self.world.device
                ),
                batch_index=env_index,
            )

            # update action range
            agent.action = Action(
                u_range=agent_max_v,
                u_multiplier = 1.0,
                u_noise = None,
                u_rot_range = 0.0,
                u_rot_multiplier = 1.0,
                u_rot_noise = None,
            )

            agent.action.to(self.world.device)  
            agent.action.batch_dim = self.world.batch_dim      
            agent.shape = Sphere(radius=agent_radius)
    
    def reset_world_at(self, env_index: int = None):
        # TODO [Shalin]: env_index is randomly not None during training, which breaks resetting
        env_index = None

        # store positions occupied by obstacles
        occupied_obstacles = torch.zeros((self.world.batch_dim, len(self.world.landmarks[self.n_agents:]), 2), 
                                         dtype=torch.float32,
                                         device=self.world.device)

        # NOTE: this needs to happen every reset to be compatible with BaseScenario reset
        for i, pos in enumerate(torch.nonzero(self.map == 1)):
            self.world.landmarks[i+self.n_agents].set_pos(
                torch.tensor(
                    [pos[1], pos[0]],
                    dtype=torch.float32,
                    device=self.device
                ),
                batch_index = env_index
            )
            occupied_obstacles[:, i] = self.world.landmarks[i+self.n_agents].state.pos

        # set agent start poses
        if self.start_poses:
            # if start poses specified in xml, use those when resetting
            for i, agent in enumerate(self.world.agents):
                agent.set_pos(
                    torch.tensor(
                        self.start_poses[i],
                        dtype=torch.float32,
                        device=self.world.device
                    ),
                    batch_index=env_index
                )
        elif self.map_name == "alcove":
            # spawn first agent position in left side of map (with noise)
            self.world.agents[0].set_pos(
                torch.tensor(
                    [
                        self.x_bounds[0] + self.obstacle_dim + 2*self.agent_radius, 
                        (self.y_bounds[1] - self.obstacle_dim) / 2
                    ],
                    dtype=torch.float32,
                    device=self.world.device,
                )
                + torch.zeros(
                    self.world.dim_p,
                    device=self.world.device,
                ).uniform_(
                    -0.02,
                    0.02,
                ),
                batch_index=env_index
            )

            # spawn second agent position in right side of map (with noise)
            self.world.agents[1].set_pos(
                torch.tensor(
                    [
                        self.x_bounds[1] - 2*self.obstacle_dim - 2*self.agent_radius,
                        (self.y_bounds[1] - self.obstacle_dim) / 2
                    ],
                    dtype=torch.float32,
                    device=self.world.device,
                )
                + torch.zeros(
                    self.world.dim_p,
                    device=self.world.device,
                ).uniform_(
                    -0.02,
                    0.02,
                ),
                batch_index=env_index
            )
        else:
            # spawn agents randomly within map
            ScenarioUtils.spawn_entities_randomly(
                self.world.agents,
                self.world,
                env_index,
                self.min_distance_between_entities,
                self.x_bounds,
                self.y_bounds,
                occupied_positions=occupied_obstacles
            )

        # store positions occupied by agents
        occupied_agents = torch.stack(
            [agent.state.pos for agent in self.world.agents], dim=1
        )

        # get all occupied positions before setting goal_poses
        occupied_positions = torch.concat(
            [occupied_obstacles, occupied_agents], dim=1
        )
        if env_index is not None:
            occupied_positions = occupied_positions[env_index].unsqueeze(0)

        # set agent goal poses
        goal_poses = []
        if self.goal_poses:
            # if goal poses specified in xml, use those when resetting
            for i, agent in enumerate(self.world.agents):
                goal_poses.append(
                    torch.tensor(
                        self.goal_poses[i],
                        dtype=torch.float32,
                        device=self.world.device
                    )
                )
        elif self.map_name == "alcove" or self.map_name == "swap":
            # shift poses by 1 so agents are swapping with another agent pose
            for i in range(self.n_agents):
                goal_poses.append(self.world.agents[(i+1) % self.n_agents].state.pos)
        else:
            # randomly generate goal poses
            for _ in self.world.agents:
                position = ScenarioUtils.find_random_pos_for_entity(
                    occupied_positions=occupied_positions,
                    env_index=env_index,
                    world=self.world,
                    min_dist_between_entities=self.min_distance_between_entities,
                    x_bounds=self.x_bounds,
                    y_bounds=self.y_bounds,
                )
                goal_poses.append(position.squeeze(1))
                occupied_positions = torch.cat([occupied_positions, position], dim=1)

        # reset the capabilities of agents
        if self.capability_aware: 
            self._reset_capability_at(env_index)

        for i, agent in enumerate(self.world.agents):
            if self.split_goals:
                goal_index = int(i // self.agents_with_same_goal)
            else:
                goal_index = 0 if i < self.agents_with_same_goal else i

            # set goal poses
            agent.goal.set_pos(goal_poses[goal_index], batch_index=env_index)

            if env_index is None:
                agent.pos_shaping = (
                    torch.linalg.vector_norm(
                        agent.state.pos - agent.goal.state.pos,
                        dim=1,
                    )
                    * self.pos_shaping_factor
                )
            else:
                agent.pos_shaping[env_index] = (
                    torch.linalg.vector_norm(
                        agent.state.pos[env_index] - agent.goal.state.pos[env_index]
                    )
                    * self.pos_shaping_factor
                )

    def reward(self, agent: Agent):
        is_first = agent == self.world.agents[0]

        if is_first:
            self.pos_rew[:] = 0
            self.final_rew[:] = 0

            for a in self.world.agents:
                self.pos_rew += self.agent_reward(a)
                a.agent_collision_rew[:] = 0

            self.all_goal_reached = torch.all(
                torch.stack([a.on_goal for a in self.world.agents], dim=-1), dim=-1
            )

            self.final_rew[self.all_goal_reached] = self.final_reward

            # check for agent-agent collisions
            for i, a in enumerate(self.world.agents):
                for j, b in enumerate(self.world.agents):
                    if i <= j:
                        continue
                    if self.world.collides(a, b):
                        distance = self.world.get_distance(a, b)
                        a.agent_collision_rew[
                            distance <= self.min_collision_distance
                        ] += self.agent_agent_collision_penalty
                        b.agent_collision_rew[
                            distance <= self.min_collision_distance
                        ] += self.agent_agent_collision_penalty

            # check for agent-obstacle collisions
            for i, a in enumerate(self.world.agents):
                for j, b in enumerate(self.world.landmarks[self.n_agents:]):
                    if self.world.collides(a, b):
                        distance = self.world.get_distance(a, b)
                        a.agent_collision_rew[
                            distance <= self.min_collision_distance
                        ] += self.agent_obstacle_collision_penalty

        pos_reward = self.pos_rew if self.shared_rew else agent.pos_rew
        return pos_reward + self.final_rew + agent.agent_collision_rew

    def agent_reward(self, agent: Agent):
        agent.distance_to_goal = torch.linalg.vector_norm(
            agent.state.pos - agent.goal.state.pos,
            dim=-1,
        )
        agent.on_goal = agent.distance_to_goal < agent.goal.shape.radius

        pos_shaping = agent.distance_to_goal * self.pos_shaping_factor
        agent.pos_rew = agent.pos_shaping - pos_shaping
        agent.pos_shaping = pos_shaping
        return agent.pos_rew

    def observation(self, agent: Agent):
        goal_poses = []
        if self.observe_all_goals:
            for a in self.world.agents:
                goal_poses.append(agent.state.pos - a.goal.state.pos)
        else:
            goal_poses.append(agent.state.pos - agent.goal.state.pos)
        return torch.cat(
            [
                agent.state.pos,
                agent.state.vel,
            ]
            + goal_poses
            + (
                [agent.sensors[0]._max_range - agent.sensors[0].measure()]
                if self.collisions
                else []
            ),
            + agent.state.capability if self.capability_aware else [],
            dim=-1,
        )

    def done(self):
        for agent in self.world.agents:
            agent.distance_to_goal = torch.linalg.vector_norm(
                agent.state.pos - agent.goal.state.pos,
                dim=-1,
            )
            agent.on_goal = agent.distance_to_goal < agent.goal.shape.radius

        self.all_goal_reached = torch.all(
            torch.stack([a.on_goal for a in self.world.agents], dim=-1), dim=-1
        )
        return self.all_goal_reached

    def info(self, agent: Agent) -> Dict[str, Tensor]:
        return {
            "pos_rew": self.pos_rew if self.shared_rew else agent.pos_rew,
            "final_rew": self.final_rew,
            "agent_collisions": agent.agent_collision_rew,
        }

    def extra_render(self, env_index: int = 0) -> "List[Geom]":
        from vmas.simulator import rendering

        geoms: List[Geom] = []

        # Communication lines
        for i, agent1 in enumerate(self.world.agents):
            for j, agent2 in enumerate(self.world.agents):
                if j <= i:
                    continue
                agent_dist = torch.linalg.vector_norm(
                    agent1.state.pos - agent2.state.pos, dim=-1
                )
                if agent_dist[env_index] <= self.comms_range:
                    color = Color.BLACK.value
                    line = rendering.Line(
                        (agent1.state.pos[env_index]),
                        (agent2.state.pos[env_index]),
                        width=1,
                    )
                    xform = rendering.Transform()
                    line.add_attr(xform)
                    line.set_color(*color)
                    geoms.append(line)

        return geoms


if __name__ == "__main__":
    render_interactively(
        "navigation_obstacles",
        control_two_agents=True,
    )
