#  Copyright (c) 2022-2023.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.
import typing
from typing import Dict, Callable, List

import torch
from torch import Tensor

import numpy as np

from vmas import render_interactively

from vmas.simulator.core import Agent, Landmark, World, Sphere, Box, Entity
from vmas.simulator.heuristic_policy import BaseHeuristicPolicy
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.sensors import Lidar
from vmas.simulator.utils import Color, ScenarioUtils, X, Y
from vmas.simulator.dynamics.waypoint_tracker import WaypointTracker
from vmas.simulator.controllers.velocity_controller import VelocityController
from vmas.simulator.dynamics.diff_drive import DiffDrive
from vmas.simulator.dynamics.holonomic import Holonomic
from vmas.simulator.dynamics.kinematic_bicycle import KinematicBicycle
from copy import deepcopy


if typing.TYPE_CHECKING:
    from vmas.simulator.rendering import Geom

DYNAMIC_MODELS = {'holonomic': Holonomic, 
                  'differential': DiffDrive,
                  'bicycle': KinematicBicycle}

class Scenario(BaseScenario):
    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        self.plot_grid = False
        self.collisions = kwargs.get("collisions", True)

        self.agents_with_same_goal = kwargs.get("agents_with_same_goal", 1)
        self.split_goals = kwargs.get("split_goals", False)
        self.observe_all_goals = kwargs.get("observe_all_goals", False)

        self.lidar_range = kwargs.get("lidar_range", 0.35)
        self.agent_radius = kwargs.get("agent_radius", 0.1)
        self.comms_range = kwargs.get("comms_range", 0)

        self.shared_rew = kwargs.get("shared_rew", True)
        self.pos_shaping_factor = kwargs.get("pos_shaping_factor", 1)
        self.final_reward = kwargs.get("final_reward", 0.01)

        self.agent_collision_penalty = kwargs.get("agent_collision_penalty", -1)

        self.min_distance_between_entities = self.agent_radius * 2 + 0.05
        self.world_semidim = 1
        self.min_collision_distance = 0.005

        # Build all of the agents from a 'robots' file. Each robot shall have
        # an ID (binary), a motion model, maximum speed (dynamics profile), and a 
        # geometry profile. The function should receive the robots file as a dictionary
        self.n_agents = kwargs.get("n_agents", 3)
        self.robots_file = kwargs.get("robots",
                                # default 3 agents
                                 {'name': 'robots_0', # name of the robots file/pool
                                  'robots':
                                      # list of robots in the robot pool
                                      [ 
                                          {'id': "001", 'dynamics': 'bicycle'},
                                          {'id': "010", 'dynamics': 'differential'},
                                          {'id': "100", 'dynamics': 'holonomic'}
                                      ]
                                  })

        assert 1 <= self.agents_with_same_goal <= self.n_agents
        if self.agents_with_same_goal > 1:
            assert (
                not self.collisions
            ), "If agents share goals they cannot be collidables"
        # agents_with_same_goal == n_agents: all agent same goal
        # agents_with_same_goal = x: the first x agents share the goal
        # agents_with_same_goal = 1: all independent goals
        if self.split_goals:
            assert (
                self.n_agents % 2 == 0
                and self.agents_with_same_goal == self.n_agents // 2
            ), "Splitting the goals is allowed when the agents are even and half the team has the same goal"

        # Make world
        world = World(batch_dim, device, substeps=2)

        entity_filter_agents: Callable[[Entity], bool] = lambda e: isinstance(e, Agent)

        # Set color to be based on kinematic model
        #   red = holonomic
        #   green = differential
        #   blue = bicycle.
        motion_model_colors_hsl = {
            'holonomic': [1,0,0],
            'differential': [0,1,0],
            'bicycle': [0,0,1]
        }
        # Unpack the robot pool
        self.robots = self.robots_file['robots'] # list of dictionaries
        
        # Add agents
        self.agent_list = []
        for i, robot in enumerate(self.robots):
            agent_id = robot['id']
            agent_dynamics = robot['dynamics']
            sensors=[Lidar(world,
                        n_rays=12,
                        max_range=self.lidar_range,
                        entity_filter=entity_filter_agents)]
            
            color = motion_model_colors_hsl[agent_dynamics]

            if agent_dynamics == 'holonomic':
                agent = Agent(
                    name=f"agent_holo-{agent_id}",
                    collide=True,
                    color=color,
                    shape=Sphere(0.1),
                    render_action=True,
                    u_range=[1, 1],
                    u_multiplier=[1, 0.001],
                    dynamics=Holonomic(),
                    sensors=sensors,
                )
                # Holonomic dynamics by default have actions
                # as force
                controller_params = [0.2, 0.6, 0.0002]
                agent.controller = VelocityController(
                agent, world, controller_params, "standard"
            )
                
            elif agent_dynamics == 'differential':
                agent = Agent(
                    name=f"agent_diff-{agent_id}",
                    collide=True,
                    color=color,
                    shape=Sphere(0.1),
                    render_action=True,
                    u_range=[1, 1],
                    u_multiplier=[1, 0.001],
                    dynamics=DiffDrive(world, integration="rk4"),
                    sensors=sensors,
                )
            elif agent_dynamics == 'bicycle':
                # width, l_f, l_r = 0.1, 0.1, 0.1
                width, l_f, l_r = robot.get('width'), robot.get('l_f'), robot.get('l_r')
                max_steering_angle = torch.deg2rad(torch.tensor(40.0))
                agent = Agent(
                    name=f"agent_bicycle-{agent_id}",
                    shape=Box(length=l_f + l_r, width=width),
                    collide=True,
                    color=color,
                    render_action=True,
                    u_range=[1, 1],
                    u_multiplier=[1, max_steering_angle],
                    dynamics=KinematicBicycle(
                        world,
                        width=width,
                        l_f=l_f,
                        l_r=l_r,
                        max_steering_angle=max_steering_angle,
                        integration="euler",  # one of "euler", "rk4"
                    ),
                    sensors=sensors,
                )
            else:
                raise ValueError(f"Undefined agent dynamics {agent_dynamics}")

            agent.pos_rew = torch.zeros(batch_dim, device=device)
            agent.agent_collision_rew = agent.pos_rew.clone()


            # Add goals
            goal = Landmark(
                name=f"goal_{agent_id}",
                collide=False,
                color=color,
            )

            world.add_landmark(goal)
            agent.goal = goal
            world.add_agent(agent)
            self.agent_list.append(agent)

        self.pos_rew = torch.zeros(batch_dim, device=device)
        self.final_rew = self.pos_rew.clone()

        return world

    def reset_world_at(self, env_index: int = None):
        ScenarioUtils.spawn_entities_randomly(
            self.world.agents,
            self.world,
            env_index,
            self.min_distance_between_entities,
            (-self.world_semidim, self.world_semidim),
            (-self.world_semidim, self.world_semidim),
        )

        occupied_positions = torch.stack(
            [agent.state.pos for agent in self.world.agents], dim=1
        )
        if env_index is not None:
            occupied_positions = occupied_positions[env_index].unsqueeze(0)

        goal_poses = []
        for _ in self.world.agents:
            position = ScenarioUtils.find_random_pos_for_entity(
                occupied_positions=occupied_positions,
                env_index=env_index,
                world=self.world,
                min_dist_between_entities=self.min_distance_between_entities,
                x_bounds=(-self.world_semidim, self.world_semidim),
                y_bounds=(-self.world_semidim, self.world_semidim),
            )
            goal_poses.append(position.squeeze(1))
            occupied_positions = torch.cat([occupied_positions, position], dim=1)
        # print(self.world.agents)
        for i, agent in enumerate(self.world.agents):
            if self.split_goals:
                goal_index = int(i // self.agents_with_same_goal)
            else:
                goal_index = 0 if i < self.agents_with_same_goal else i

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

            for i, a in enumerate(self.world.agents):
                for j, b in enumerate(self.world.agents):
                    if i <= j:
                        continue
                    if self.world.collides(a, b):
                        distance = self.world.get_distance(a, b)
                        a.agent_collision_rew[
                            distance <= self.min_collision_distance
                        ] += self.agent_collision_penalty
                        b.agent_collision_rew[
                            distance <= self.min_collision_distance
                        ] += self.agent_collision_penalty

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
                agent.state.rot,
                agent.state.vel,
            ]
            + goal_poses
            + (
                [agent.sensors[0]._max_range - agent.sensors[0].measure()]
                if self.collisions
                else []
            ),
            dim=-1,
        )

    def done(self):
        return torch.stack(
            [
                torch.linalg.vector_norm(
                    agent.state.pos - agent.goal.state.pos,
                    dim=-1,
                )
                < (agent.shape.radius if isinstance(agent.shape, Sphere) else agent.shape.width)
                for agent in self.world.agents
            ],
            dim=-1,
        ).all(-1)

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

class HeuristicPolicy(BaseHeuristicPolicy):
    def compute_action(self, observation: torch.Tensor, u_range: float) -> torch.Tensor:
        assert self.continuous_actions

        current_pos = observation[:, :2]
        current_vel = observation[:, 2:4]
        # NOTE: this assumes agents can only observe own goal (observe_all_goals = False)
        dir_from_goal = observation[:, 4:6]
        # this is a weird output, it's not the raw LIDAR but rather
        # LIDAR max_range - LIDAR measurement * n_lidar_rays
        lidar = observation[:, 6:]

        # move towards goal (w/out considering obstacles) 
        action_dir = -dir_from_goal

        # however, if any obstacles in visibility range, move directly away
        # from them, instead of towards goal
        # TODO: implement? performance is pretty good for this env though
        # visible_objects = torch.any(lidar > 0.0)
        # if visible_objects:
        #     # TODO: instead of finding max dist, should find min of > 0.0
        #     _, object_dir_index = torch.max(lidar, dim=1)
        #     angle_to_obj = object_dir_index / lidar.shape[1] * 2 * torch.pi
        #     dir_to_obj = [torch.cos(angle_to_obj), torch.sin(angle_to_obj)]
        #     towards_obj_dir = torch.stack(dir_to_obj, dim=1)
        #     action_dir = -0.01 * towards_obj_dir

        action = torch.clamp(
            action_dir,
            min=-u_range,
            max=u_range,
        )

        return action

class HeuristicPolicy(BaseHeuristicPolicy):
    def __init__(self, clf_epsilon = 0.2, clf_slack = 100.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.clf_epsilon = clf_epsilon  # Exponential CLF convergence rate
        self.clf_slack = clf_slack  # weights on CLF-QP slack variable

    def compute_action(self, observation: Tensor, u_range: Tensor) -> Tensor:
        """
        QP inputs:
        These values need to computed apriri based on observation before passing into QP

        V: Lyapunov function value
        lfV: Lie derivative of Lyapunov function
        lgV: Lie derivative of Lyapunov function
        CLF_slack: CLF constraint slack variable

        QP outputs:
        u: action
        CLF_slack: CLF constraint slack variable, 0 if CLF constraint is satisfied
        """
        # Install it with: pip install cvxpylayers
        import cvxpy as cp
        from cvxpylayers.torch import CvxpyLayer

        self.n_env = observation.shape[0]
        self.device = observation.device
        agent_pos = observation[:, :2]
        agent_vel = observation[:, 2:4]
        goal_pos = (-1.0) * (observation[:, 4:6] - agent_pos)

        # Pre-compute tensors for the CLF and CBF constraints,
        # Lyapunov Function from: https://arxiv.org/pdf/1903.03692.pdf

        # Laypunov function
        V_value = (
            (agent_pos[:, X] - goal_pos[:, X]) ** 2
            + 0.5 * (agent_pos[:, X] - goal_pos[:, X]) * agent_vel[:, X]
            + agent_vel[:, X] ** 2
            + (agent_pos[:, Y] - goal_pos[:, Y]) ** 2
            + 0.5 * (agent_pos[:, Y] - goal_pos[:, Y]) * agent_vel[:, Y]
            + agent_vel[:, Y] ** 2
        )

        LfV_val = (2 * (agent_pos[:, X] - goal_pos[:, X]) + agent_vel[:, X]) * (
            agent_vel[:, X]
        ) + (2 * (agent_pos[:, Y] - goal_pos[:, Y]) + agent_vel[:, Y]) * (
            agent_vel[:, Y]
        )
        LgV_vals = torch.stack(
            [
                0.5 * (agent_pos[:, X] - goal_pos[:, X]) + 2 * agent_vel[:, X],
                0.5 * (agent_pos[:, Y] - goal_pos[:, Y]) + 2 * agent_vel[:, Y],
            ],
            dim=1,
        )
        # Define Quadratic Program (QP) based controller
        u = cp.Variable(2)
        V_param = cp.Parameter(1)  # Lyapunov Function: V(x): x -> R, dim: (1,1)
        lfV_param = cp.Parameter(1)
        lgV_params = cp.Parameter(
            2
        )  # Lie derivative of Lyapunov Function, dim: (1, action_dim)
        clf_slack = cp.Variable(1)  # CLF constraint slack variable, dim: (1,1)

        constraints = []

        # QP Cost F = u^T @ u + clf_slack**2
        qp_objective = cp.Minimize(cp.sum_squares(u) + self.clf_slack * clf_slack**2)

        # control bounds between u_range
        constraints += [u <= u_range]
        constraints += [u >= -u_range]
        # CLF constraint
        constraints += [
            lfV_param + lgV_params @ u + self.clf_epsilon * V_param + clf_slack <= 0
        ]

        QP_problem = cp.Problem(qp_objective, constraints)

        # Initialize CVXPY layers
        QP_controller = CvxpyLayer(
            QP_problem, parameters=[V_param, lfV_param, lgV_params], variables=[u]
        )

        # Solve QP
        CVXpylayer_parameters = [V_value.unsqueeze(1), LfV_val.unsqueeze(1), LgV_vals]
        action = QP_controller(*CVXpylayer_parameters, solver_args={"max_iters": 500})[
            0
        ]

        return action


if __name__ == "__main__":
    render_interactively(
        __file__,
        control_two_agents=True,
    )
