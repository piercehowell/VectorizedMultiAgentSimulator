#  Copyright (c) 2022-2024.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.

import random
import torch
import numpy as np

from vmas import render_interactively
from vmas.simulator.core import Agent, Box, Landmark, Sphere, World
from vmas.simulator.heuristic_policy import BaseHeuristicPolicy
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.utils import Color, ScenarioUtils

from typing import Dict, Callable, List
from torch import Tensor

import typing
if typing.TYPE_CHECKING:
    from vmas.simulator.rendering import Geom


class Scenario(BaseScenario):
    def get_rng_state(self, device):
        """
        Returns a tuple of the form
        (numpy random state, python's random state, torch's random state, torch.cuda's random state)
        """
        np_rng_state = np.random.get_state()
        py_rng_state = random.getstate()
        torch_rng_state = torch.get_rng_state()
        torch_cuda_rng_state = torch.cuda.get_rng_state(device)

        return (np_rng_state, py_rng_state, torch_rng_state, torch_cuda_rng_state)

    def set_eval_seed(self, eval_seed):
        """
        Set a new seed for numpy, python.random, torch.random, and torch.cuda.random.

        Intended to be used only with eval_seed + wrapped by get/set_rng_state().
        """
        torch.manual_seed(self.eval_seed)
        torch.cuda.manual_seed_all(self.eval_seed)
        random.seed(self.eval_seed)
        np.random.seed(self.eval_seed)

    def set_rng_state(self, old_rng_state, device):
        """
        Restore the prior RNG state (based on the return value of get_rng_state).
        """
        assert old_rng_state is not None, "set_rng_state() must be called with the return value of get_rng_state()!"

        np_rng_state, py_rng_state, torch_rng_state, torch_cuda_rng_state = old_rng_state 

        np.random.set_state(np_rng_state)
        random.setstate(py_rng_state)
        torch.set_rng_state(torch_rng_state)
        torch.cuda.set_rng_state(torch_cuda_rng_state, device)

    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        # if passed in, we will use an "eval seed" independent of the training seed for VMAS only
        self.eval_seed = kwargs.get("eval_seed", None)

        n_agents = kwargs.get("n_agents", 4)

        # packages
        self.n_packages = kwargs.get("n_packages", 1)
        self.package_width = kwargs.get("package_width", 0.15)
        self.package_length = kwargs.get("package_length", 0.15)
        self.package_rotatable = kwargs.get("package_rotatable", True)

        # partial obs
        self.partial_observations = kwargs.get("partial_observations", True)
        self.package_observation_radius = kwargs.get("package_observation_radius", 0.35)

        # realism
        self.linear_friction = kwargs.get("linear_friction", 0.1)
        self.angular_friction = kwargs.get("angular_friction", 0.1)
        self.drag = kwargs.get("drag", 0.0)
        self.package_mass = kwargs.get("package_mass", 50)
        # TODO: implement automated domain randomization here?

        # rewards
        self.agent_package_dist_reward_factor = kwargs.get("agent_package_dist_reward_factor", 0.1)
        self.package_goal_dist_reward_factor = kwargs.get("package_goal_dist_reward_factor", 100)
        self.interagent_collision_penalty = kwargs.get("interagent_collision_penalty", -1)
        assert self.interagent_collision_penalty <= 0, f"self.interagent_collision_penalty must be <= 0, current value is {self.interagent_collision_penalty}!"
        self.add_dense_reward = kwargs.get("add_dense_reward", True)
        self.package_on_goal_reward_factor = kwargs.get("package_on_goal_reward_factor", 1.0)

        # capabilities
        self.capability_mult_range = kwargs.get("capability_mult_range", [0.5, 2])
        self.capability_mult_min = self.capability_mult_range[0]
        self.capability_mult_max = self.capability_mult_range[1]
        self.capability_representation = kwargs.get("capability_representation", "raw")

        # general world settings
        self.world_semidim = 0.75 
        self.default_agent_radius = 0.03
        self.default_agent_u = 0.6
        self.default_agent_mass = 1.0
        self.min_collision_distance = 0.05 * self.default_agent_radius # default navigation collision dist is 5% of the agent radius

        rng_state = None
        if self.eval_seed:
            rng_state = self.get_rng_state(device)
            self.set_eval_seed(self.eval_seed)

        # Make world
        world = World(
            batch_dim,
            device,
            x_semidim=self.world_semidim
            + 2 * self.default_agent_radius
            + max(self.package_length, self.package_width),
            y_semidim=self.world_semidim
            + 2 * self.default_agent_radius
            + max(self.package_length, self.package_width),
            linear_friction=self.linear_friction,
            angular_friction=self.angular_friction,
            drag=self.drag,
        )

        # Add agents
        capabilities = [] # save capabilities for relative capabilities later
        for i in range(n_agents):
            u_mult = self.default_agent_u * random.uniform(self.capability_mult_min, self.capability_mult_max)
            radius = self.default_agent_radius * random.uniform(self.capability_mult_min, self.capability_mult_max)
            mass = self.default_agent_mass * random.uniform(self.capability_mult_min, self.capability_mult_max)

            capabilities.append([u_mult, radius, mass])

            agent = Agent(
                name=f"agent_{i}", 
                u_multiplier=u_mult,
                shape=Sphere(radius),
                mass=mass,
            )
            agent.agent_collision_rew = torch.zeros(batch_dim, device=device)

            # TODO: add VelocityController!
            # TODO: diff drive?
            # controller_params = [0.2, 0.6, 0.0002]
            # agent.controller = VelocityController(
            #     agent, world, controller_params, "standard"
            # )

            world.add_agent(agent)

        self.capabilities = torch.tensor(capabilities)

        # Add landmarks
        goal = Landmark(
            name="goal",
            collide=False,
            shape=Sphere(radius=0.15),
            color=Color.LIGHT_GREEN,
        )
        world.add_landmark(goal)
        self.packages = []
        for i in range(self.n_packages):
            package = Landmark(
                name=f"package {i}",
                collide=True,
                movable=True,
                rotatable=self.package_rotatable,
                mass=self.package_mass,
                shape=Box(length=self.package_length, width=self.package_width),
                color=Color.RED,
            )
            package.goal = goal
            self.packages.append(package)
            world.add_landmark(package)

        if self.eval_seed:
            self.set_rng_state(rng_state, device)

        return world

    def reset_world_at(self, env_index: int = None):
        rng_state = None
        if self.eval_seed:
            rng_state = self.get_rng_state(self.world.device)
            self.set_eval_seed(self.eval_seed)

        # only do this during batched resets!
        if not env_index:        
            capabilities = [] # save capabilities for relative capabilities later
            for agent in self.world.agents:
                u_mult = self.default_agent_u * random.uniform(self.capability_mult_min, self.capability_mult_max)
                radius = self.default_agent_radius * random.uniform(self.capability_mult_min, self.capability_mult_max)
                mass = self.default_agent_mass * random.uniform(self.capability_mult_min, self.capability_mult_max)

                capabilities.append([u_mult, radius, mass])

                agent.u_multiplier=u_mult
                agent.shape=Sphere(radius)
                agent.mass=mass

            self.capabilities = torch.tensor(capabilities)

        # spawn goal at origin
        goal = self.world.landmarks[0]
        goal.state.pos = torch.zeros(goal.state.pos.shape, device=self.world.device)
        goal_occupied_pos = torch.stack(
            [goal.state.pos], dim=1
        )
        if env_index is not None:
            goal_occupied_pos = goal_occupied_pos[env_index].unsqueeze(0)

        # then spawn packages randomly around it
        ScenarioUtils.spawn_entities_randomly(
            self.packages,
            self.world,
            env_index,
            min_dist_between_entities=max(
                package.shape.circumscribed_radius() + goal.shape.radius + 0.01
                for package in self.packages
            ),
            x_bounds=(
                -self.world_semidim,
                self.world_semidim,
            ),
            y_bounds=(
                -self.world_semidim,
                self.world_semidim,
            ),
            occupied_positions=goal_occupied_pos,
        )

        package_occupied_pos = torch.stack(
            [package.state.pos for package in self.packages], dim=1
        )
        if env_index is not None:
            package_occupied_pos = package_occupied_pos[env_index].unsqueeze(0)

        # then spawn agents randomly around packages
        ScenarioUtils.spawn_entities_randomly(
            self.world.agents,
            self.world,
            env_index,
            min_dist_between_entities=self.packages[0].shape.circumscribed_radius() + goal.shape.radius + 0.01,
            x_bounds=(
                -self.world_semidim,
                self.world_semidim,
            ),
            y_bounds=(
                -self.world_semidim,
                self.world_semidim,
            ),
            occupied_positions=package_occupied_pos,
        )
        
        self.package_starting_dists = []
        self.og_package_positions = []
        for i, package in enumerate(self.packages):
            package.on_goal = self.world.is_overlapping(package, package.goal)
            
            self.og_package_positions.append(package.state.pos)
            self.package_starting_dists.append(
                torch.linalg.vector_norm(package.state.pos - package.goal.state.pos, dim=1)
            )

            if env_index is None:
                package.global_shaping = (
                    torch.linalg.vector_norm(
                        package.state.pos - package.goal.state.pos, dim=1
                    )
                    * self.package_goal_dist_reward_factor
                )
            else:
                package.global_shaping[env_index] = (
                    torch.linalg.vector_norm(
                        package.state.pos[env_index] - package.goal.state.pos[env_index]
                    )
                    * self.package_goal_dist_reward_factor
                )

        if self.eval_seed:
            self.set_rng_state(rng_state, self.world.device)

    def reward(self, agent: Agent):
        # rewards under is_first only need to be applied once
        is_first = agent == self.world.agents[0]
        if is_first:
            self.rew = torch.zeros(
                self.world.batch_dim, device=self.world.device, dtype=torch.float32
            )

            # reward for how close package is to goal
            # (shared across team)
            for i, package in enumerate(self.packages):
                package.dist_to_goal = torch.linalg.vector_norm(
                    package.state.pos - package.goal.state.pos, dim=1
                )
                package.on_goal = self.world.is_overlapping(package, package.goal)
                package.color = torch.tensor(
                    Color.RED.value, device=self.world.device, dtype=torch.float32
                ).repeat(self.world.batch_dim, 1)
                package.color[package.on_goal] = torch.tensor(
                    Color.GREEN.value, device=self.world.device, dtype=torch.float32
                )

                # dense reward
                if self.add_dense_reward:
                    package_shaping = package.dist_to_goal * self.package_goal_dist_reward_factor
                    self.rew[~package.on_goal] += (
                        package.global_shaping[~package.on_goal]
                        - package_shaping[~package.on_goal]
                        )
                    package.global_shaping = package_shaping
                
                # positive reward when the agent achieves the goal
                self.rew[package.on_goal] += 1.0 * self.package_on_goal_reward_factor
                

            # penalty (negative rew) for agent-agent collisions
            for a in self.world.agents:
                a.agent_collision_rew[:] = 0

            for i, a in enumerate(self.world.agents):
                for j, b in enumerate(self.world.agents):
                    if i <= j:
                        continue
                    if self.world.collides(a, b):
                        distance = self.world.get_distance(a, b)
                        a.agent_collision_rew[
                            distance <= self.min_collision_distance
                        ] += self.interagent_collision_penalty
                        b.agent_collision_rew[
                            distance <= self.min_collision_distance
                        ] += self.interagent_collision_penalty

        # reward for how close agents are to all packages
        if self.add_dense_reward:
            for i, package in enumerate(self.packages):
                dist_to_pkg = torch.linalg.vector_norm(agent.state.pos - package.state.pos, dim=-1)
                # any small distance gets "floored"
                # dist_to_pkg[dist_to_pkg < 0.1] = 0.1

                self.rew += -dist_to_pkg * self.agent_package_dist_reward_factor

        return self.rew + agent.agent_collision_rew
    
    def info(self, agent: Agent):
        """
        Log information about agent and scenario state.

        :param agent: Agent batch to compute info of
        :return: info: A dict with a key for each info of interest, and a tensor value  of shape (n_envs, info_size)
        """
        # NOTE: we are assuming in logging that these metrics only matter at the ends of episodes
        # can fix that either in logging or in here
        dist_to_pkg = torch.zeros(self.world.batch_dim, device=self.world.device)
        dist_to_goal = torch.zeros(self.world.batch_dim, device=self.world.device)
        goal = self.world.landmarks[0]
        for i, package in enumerate(self.packages):
            dist_to_goal += torch.linalg.vector_norm(
                    package.state.pos - package.goal.state.pos, dim=1
                ) - goal.shape.radius
            dist_to_pkg += torch.linalg.vector_norm(agent.state.pos - package.state.pos, dim=-1) - agent.shape.radius

        success_rate = torch.sum(
            torch.stack(
                [package.on_goal for package in self.packages],
                dim=1,
            ),
            dim=-1
        ) / len(self.packages)

        # TODO(Kevin): double-check that agent_collision_rew works across all agents (and we're not just logging 1 agent's penalties)
        return {"dist_to_goal": dist_to_goal, "dist_to_pkg": dist_to_pkg, "success_rate": success_rate,
                "curiosity_state": self.curiosity_state(agent),
                "agent_collision_rew": agent.agent_collision_rew,
                "environment_state": self.environment_state(agent)}

    def get_capability_repr(self, agent: Agent):
        """
        Get capability representation:
            raw = raw multiplier values
            relative = zero-meaned (taking mean of team into account)
            mixed = raw + relative (concatenated)
        """
        if self.capability_representation == "raw":
            # agent's normal capabilities
            u_mult = agent.u_multiplier
            radius = agent.shape.radius
            mass = agent.mass

            capability_repr = [
                torch.tensor(
                    u_mult, device=self.world.device
                ).repeat(self.world.batch_dim, 1),
                torch.tensor(
                    radius, device=self.world.device
                ).repeat(self.world.batch_dim, 1),
                torch.tensor(
                    mass, device=self.world.device
                ).repeat(self.world.batch_dim, 1),
            ]

        elif self.capability_representation == "relative":
            # compute the mean capabilities across the team's agents
            team_mean = list(torch.mean(self.capabilities, dim=0))
            # then compute "relative capability" of this agent by subtracting the mean
            rel_u_mult = agent.u_multiplier - team_mean[0].item()
            rel_radius = agent.shape.radius - team_mean[1].item()
            rel_mass = agent.mass - team_mean[2].item()
            capability_repr = [
                torch.tensor(
                    rel_u_mult, device=self.world.device
                ).repeat(self.world.batch_dim, 1),
                torch.tensor(
                    rel_radius, device=self.world.device
                ).repeat(self.world.batch_dim, 1),
                torch.tensor(
                    rel_mass, device=self.world.device
                ).repeat(self.world.batch_dim, 1),
            ]

        elif self.capability_representation == "mixed":
            # agent's normal capabilities
            u_mult = agent.u_multiplier
            radius = agent.shape.radius
            mass = agent.mass

            # compute the mean capabilities across the team's agents
            team_mean = list(torch.mean(self.capabilities, dim=0))
            # then compute "relative capability" of this agent by subtracting the mean
            rel_u_mult = agent.u_multiplier - team_mean[0].item()
            rel_radius = agent.shape.radius - team_mean[1].item()
            rel_mass = agent.mass - team_mean[2].item()
            capability_repr = [
                torch.tensor(
                    u_mult, device=self.world.device
                ).repeat(self.world.batch_dim, 1),
                torch.tensor(
                    radius, device=self.world.device
                ).repeat(self.world.batch_dim, 1),
                torch.tensor(
                    mass, device=self.world.device
                ).repeat(self.world.batch_dim, 1),
                torch.tensor(
                    rel_u_mult, device=self.world.device
                ).repeat(self.world.batch_dim, 1),
                torch.tensor(
                    rel_radius, device=self.world.device
                ).repeat(self.world.batch_dim, 1),
                torch.tensor(
                    rel_mass, device=self.world.device
                ).repeat(self.world.batch_dim, 1),
            ]
        return capability_repr

    
    def partial_observation(self, agent: Agent):
        """
        Parital observation of the boxes for the agent
        """
         # get positions of all entities in this agent's reference frame
        package_obs = []
        out_of_obs_val = -0.0001 # default value used for out-of-observation data in the observation vector
        for i, package in enumerate(self.packages):
            # box starting position and goal position alway part of the observation
            package_obs.append(self.og_package_positions[i])
            package_obs.append(package.on_goal.unsqueeze(-1))
            
            mask = (torch.linalg.vector_norm(package.state.pos - agent.state.pos, dim=-1) < self.package_observation_radius)
            pkg_state_vec = package.state.pos.clone()
            pkg_rot_vec = package.state.rot.clone()
            pkg_vel_vec = package.state.vel.clone()
            pkg_ang_vel_vec = package.state.ang_vel.clone()
            pkg_dist_to_goal_vec = package.state.pos - package.goal.state.pos
            agent_dist_to_pkg_vec = package.state.pos - agent.state.pos
            
            pkg_state_vec[~mask] = out_of_obs_val
            pkg_rot_vec[~mask] = out_of_obs_val
            pkg_vel_vec[~mask] = out_of_obs_val
            pkg_ang_vel_vec[~mask] = out_of_obs_val
            pkg_dist_to_goal_vec[~mask] = out_of_obs_val
            agent_dist_to_pkg_vec[~mask] = out_of_obs_val

            package_obs.append(pkg_state_vec)
            package_obs.append(pkg_rot_vec)
            package_obs.append(pkg_vel_vec)
            package_obs.append(pkg_ang_vel_vec)
            package_obs.append(pkg_dist_to_goal_vec)
            package_obs.append(agent_dist_to_pkg_vec)          

        capability_repr = self.get_capability_repr(agent)

        return torch.cat(
            [
                agent.state.pos,
                agent.state.vel,
                *package_obs,
            ] + capability_repr,
            dim=-1,
        )

    def default_observation(self, agent: Agent):
        # get positions of all entities in this agent's reference frame

        package_obs = []
        for package in self.packages:
            package_obs.append(package.state.pos - package.goal.state.pos)
            # NOTE: this is the raw observed rotation, rather than relative to goal rotation (since we don't care at what angle  the pkg gets to goal)
            package_obs.append(package.state.rot)
            package_obs.append(package.state.pos - agent.state.pos)
            package_obs.append(package.state.vel)
            package_obs.append(package.state.ang_vel)
            package_obs.append(package.on_goal.unsqueeze(-1))

        capability_repr = self.get_capability_repr(agent)

        return capability_repr

    def default_observation(self, agent: Agent):
        # get positions of all entities in this agent's reference frame

        package_obs = []
        for package in self.packages:
            package_obs.append(package.state.pos - package.goal.state.pos)
            package_obs.append(package.state.pos - agent.state.pos)
            package_obs.append(package.state.vel)
            package_obs.append(package.on_goal.unsqueeze(-1))

        capability_repr = self.get_capability_repr(agent)
        return torch.cat(
            [
                agent.state.pos,
                agent.state.vel,
                *package_obs,
            ] + capability_repr,
            dim=-1,
        )
    
    def observation(self, agent: Agent):
        if self.partial_observations:
            return self.partial_observation(agent)
        else:
            return self.default_observation(agent)
    
    def environment_state(self, agent: Agent):
        """Generate the state of the entire environment. Typically used for the critic network.
        This is not meant to be the agent's observation since it contains privledged information."""

        # only compute for the first agent. save to self.env_state
        is_first_agent = (agent == self.world.agents[0])
        if is_first_agent:

            # package information
            package_obs = []
            for i, package in enumerate(self.packages):
                package_dist_from_goal_at_start = self.package_starting_dists[i]
                package_dist_to_goal = torch.linalg.vector_norm(package.state.pos - package.goal.state.pos, dim=1)
                
                package_obs += [
                    # package_dist_from_goal_at_start,
                    package.state.pos,
                    package.state.rot,
                    package.state.vel,
                    package_dist_to_goal.unsqueeze(-1),
                    package.goal.state.pos
                ]
            
            # agent information
            agent_related_obs = []
            for j, agent in enumerate(self.world.agents):

                # package_to_agent state info
                for package in self.packages:
                    package_dist_to_agent = torch.linalg.vector_norm(package.state.pos - agent.state.pos, dim=1)
                    package_to_agent_state_diff = package.state.pos - agent.state.pos
                    
                    agent_related_obs += [
                        package_dist_to_agent.unsqueeze(-1),
                        package_to_agent_state_diff
                    ]
                
                capability_repr = self.get_capability_repr(agent)
                agent_related_obs += [
                    agent.state.pos,
                    agent.state.rot,
                    agent.state.vel,
                ] + capability_repr + package_obs
                
                self.env_state = torch.cat(agent_related_obs, dim=-1)
        return self.env_state

    def curiosity_state(self, agent: Agent):
        """Curiosity state used for Random Netwok Distillation intrinsic
        reward"""
        package_obs = []
        for i, package in enumerate(self.packages):
            package_dist_from_goal_at_start = self.package_starting_dists[i]

            # normalized
            package_dist_to_goal = torch.linalg.vector_norm(package.state.pos - package.goal.state.pos, dim=1) / (package_dist_from_goal_at_start + 1e-6)
            package_dist_to_goal = torch.clamp(package_dist_to_goal, -1e-6, 1.1)
            package_dist_to_agent = torch.clamp(torch.linalg.vector_norm(package.state.pos - agent.state.pos, dim=1), 0.0, 1.0) * 0.001
            package_vel = package.state.vel
            package_obs += [
                package_dist_to_goal.unsqueeze(-1),
                package_dist_to_agent.unsqueeze(-1),
                package_vel,
            ]
        cs = torch.cat(
            [
                *package_obs
            ],
            dim=-1
        )
        # print("CS", cs)
        return cs

    def done(self):
        return torch.all(
            torch.stack(
                [package.on_goal for package in self.packages],
                dim=1,
            ),
            dim=-1,
        )
    
    def extra_render(self, env_index: int = 0) -> "List[Geom]":
        from vmas.simulator import rendering

        geoms: List[Geom] = []
        if not self.partial_observations:
            return geoms

        for i, agent in enumerate(self.world.agents):

            obs_circle = rendering.make_circle(self.package_observation_radius, filled=True)
            xform = rendering.Transform()
            xform.set_translation(*agent.state.pos[env_index])
            obs_circle.add_attr(xform)
            obs_circle.set_color(*(0.827, 0.827, 0.827, 0.65))
            geoms.append(obs_circle)
       
        return geoms

class HeuristicPolicy(BaseHeuristicPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lookahead = 0.0  # evaluate u at this value along the spline
        self.start_vel_dist_from_target_ratio = (
            0.5  # distance away from the target for the start_vel to point
        )
        self.start_vel_behind_ratio = 0.5  # component of start vel pointing directly behind target (other component is normal)
        self.start_vel_mag = 1.0  # magnitude of start_vel (determines speed along the whole trajectory, as spline is recalculated continuously)
        self.hit_vel_mag = 1.0
        self.package_radius = 0.15 / 2
        self.agent_radius = -0.02
        self.dribble_slowdown_dist = 0.0
        self.speed = 0.95

    def compute_action(self, observation: torch.Tensor, u_range: float) -> torch.Tensor:
        self.n_env = observation.shape[0]
        self.device = observation.device
        agent_pos = observation[:, :2]
        agent_vel = observation[:, 2:4]
        package_pos = observation[:, 6:8] + agent_pos
        goal_pos = -observation[:, 4:6] + package_pos
        # control = self.get_action(goal_pos, curr_pos=agent_pos, curr_vel=agent_vel)
        control = self.dribble(agent_pos, package_pos, goal_pos)
        control *= self.speed * u_range
        return torch.clamp(control, -u_range, u_range)

    def dribble(self, agent_pos, package_pos, goal_pos, agent_vel=None):
        package_disp = goal_pos - package_pos
        ball_dist = package_disp.norm(dim=-1)
        direction = package_disp / ball_dist[:, None]
        hit_pos = package_pos - direction * (self.package_radius + self.agent_radius)
        hit_vel = direction * self.hit_vel_mag
        start_vel = self.get_start_vel(
            hit_pos, hit_vel, agent_pos, self.start_vel_mag * 2
        )
        slowdown_mask = ball_dist <= self.dribble_slowdown_dist
        hit_vel[slowdown_mask, :] *= (
            ball_dist[slowdown_mask, None] / self.dribble_slowdown_dist
        )
        return self.get_action(
            target_pos=hit_pos,
            target_vel=hit_vel,
            curr_pos=agent_pos,
            curr_vel=agent_vel,
            start_vel=start_vel,
        )

    def hermite(self, p0, p1, p0dot, p1dot, u=0.0, deriv=0):
        # Formatting
        u = u.reshape((-1,))

        # Calculation
        U = torch.stack(
            [
                self.nPr(3, deriv) * (u ** max(0, 3 - deriv)),
                self.nPr(2, deriv) * (u ** max(0, 2 - deriv)),
                self.nPr(1, deriv) * (u ** max(0, 1 - deriv)),
                self.nPr(0, deriv) * (u**0),
            ],
            dim=1,
        ).float()
        A = torch.tensor(
            [
                [2.0, -2.0, 1.0, 1.0],
                [-3.0, 3.0, -2.0, -1.0],
                [0.0, 0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
            ],
            device=U.device,
        )
        P = torch.stack([p0, p1, p0dot, p1dot], dim=1)
        ans = U[:, None, :] @ A[None, :, :] @ P
        ans = ans.squeeze(1)
        return ans

    def nPr(self, n, r):
        if r > n:
            return 0
        ans = 1
        for k in range(n, max(1, n - r), -1):
            ans = ans * k
        return ans

    def get_start_vel(self, pos, vel, start_pos, start_vel_mag):
        start_vel_mag = torch.as_tensor(start_vel_mag, device=self.device).view(-1)
        goal_disp = pos - start_pos
        goal_dist = goal_disp.norm(dim=-1)
        vel_mag = vel.norm(dim=-1)
        vel_dir = vel.clone()
        vel_dir[vel_mag > 0] /= vel_mag[vel_mag > 0, None]
        goal_dir = goal_disp / goal_dist[:, None]

        vel_dir_normal = torch.stack([-vel_dir[:, 1], vel_dir[:, 0]], dim=1)
        dot_prod = (goal_dir * vel_dir_normal).sum(dim=1)
        vel_dir_normal[dot_prod > 0, :] *= -1

        dist_behind_target = self.start_vel_dist_from_target_ratio * goal_dist
        point_dir = -vel_dir * self.start_vel_behind_ratio + vel_dir_normal * (
            1 - self.start_vel_behind_ratio
        )

        target_pos = pos + point_dir * dist_behind_target[:, None]
        target_disp = target_pos - start_pos
        target_dist = target_disp.norm(dim=1)
        start_vel_aug_dir = target_disp
        start_vel_aug_dir[target_dist > 0] /= target_dist[target_dist > 0, None]
        start_vel = start_vel_aug_dir * start_vel_mag[:, None]
        return start_vel

    def get_action(
        self,
        target_pos,
        target_vel=None,
        start_pos=None,
        start_vel=None,
        curr_pos=None,
        curr_vel=None,
    ):
        if curr_pos is None:  # If None, target_pos is assumed to be a relative position
            curr_pos = torch.zeros(target_pos.shape, device=self.device)
        if curr_vel is None:  # If None, curr_vel is assumed to be 0
            curr_vel = torch.zeros(target_pos.shape, device=self.device)
        if (
            start_pos is None
        ):  # If None, start_pos is assumed to be the same as curr_pos
            start_pos = curr_pos
        if target_vel is None:  # If None, target_vel is assumed to be 0
            target_vel = torch.zeros(target_pos.shape, device=self.device)
        if start_vel is None:  # If None, start_vel is calculated with get_start_vel
            start_vel = self.get_start_vel(
                target_pos, target_vel, start_pos, self.start_vel_mag * 2
            )

        u_start = torch.ones(curr_pos.shape[0], device=self.device) * self.lookahead
        des_curr_pos = self.hermite(
            start_pos,
            target_pos,
            start_vel,
            target_vel,
            u=u_start,
            deriv=0,
        )
        des_curr_vel = self.hermite(
            start_pos,
            target_pos,
            start_vel,
            target_vel,
            u=u_start,
            deriv=1,
        )
        des_curr_pos = torch.as_tensor(des_curr_pos, device=self.device)
        des_curr_vel = torch.as_tensor(des_curr_vel, device=self.device)
        control = 0.5 * (des_curr_pos - curr_pos) + 0.5 * (des_curr_vel - curr_vel)
        return control


if __name__ == "__main__":
    render_interactively(__file__, control_two_agents=True)
