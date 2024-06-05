#  Copyright (c) 2022-2023.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.

import numpy as np
import random
import torch

from vmas import render_interactively
from vmas.simulator.core import Agent, Landmark, Sphere, World, Line, Box
from vmas.simulator.heuristic_policy import BaseHeuristicPolicy
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.utils import Color, Y


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
        self.n_agents = kwargs.get("n_agents", 3)
        self.package_mass = kwargs.get("package_mass", 5)
        self.random_package_pos_on_line = kwargs.get("random_package_pos_on_line", True)
        self.world_semidim = kwargs.get("world_semidim", 1.0)
        self.gravity = kwargs.get("gravity", -0.05)
        self.eval_seed = kwargs.get("eval_seed", None)

        # capabilities
        self.capability_mult_range = kwargs.get("capability_mult_range", [0.75, 1.25])
        self.multiple_ranges = kwargs.get("multiple_ranges", False)
        if not self.multiple_ranges:
            self.capability_mult_min = self.capability_mult_range[0]
            self.capability_mult_max = self.capability_mult_range[1]
        self.capability_representation = kwargs.get("capability_representation", "raw")
        self.default_u_multiplier = kwargs.get("default_u_multiplier", 0.7)
        self.default_agent_radius = kwargs.get("default_agent_radius", 0.03)
        self.default_agent_mass = kwargs.get("default_agent_mass", 1)

        # metrics
        self.success_rate = None

        # rng
        rng_state = None
        if self.eval_seed:
            rng_state = self.get_rng_state(device)
            self.set_eval_seed(self.eval_seed)

        assert self.n_agents > 1

        self.line_length = 0.8

        self.shaping_factor = 1
        self.fall_reward = -0.1

        # Make world
        world = World(batch_dim, device, gravity=(0.0, self.gravity), y_semidim=self.world_semidim)
        # Add agents
        capabilities = [] # save capabilities for relative capabilities later
        for i in range(self.n_agents):
            if self.multiple_ranges:
                cap_idx = int(random.choice(np.arange(len(self.capability_mult_range))))
                self.capability_mult_min = self.capability_mult_range[cap_idx][0]
                self.capability_mult_max = self.capability_mult_range[cap_idx][1]
                print("MADE IT HERE")
            max_u = self.default_u_multiplier * random.uniform(self.capability_mult_min, self.capability_mult_max)
            if self.multiple_ranges:
                cap_idx = int(random.choice(np.arange(len(self.capability_mult_range))))
                self.capability_mult_min = self.capability_mult_range[cap_idx][0]
                self.capability_mult_max = self.capability_mult_range[cap_idx][1]
            radius = self.default_agent_radius * random.uniform(self.capability_mult_min, self.capability_mult_max)
            if self.multiple_ranges:
                cap_idx = int(random.choice(np.arange(len(self.capability_mult_range))))
                self.capability_mult_min = self.capability_mult_range[cap_idx][0]
                self.capability_mult_max = self.capability_mult_range[cap_idx][1]
            mass = self.default_agent_mass * random.uniform(self.capability_mult_min, self.capability_mult_max)

            agent = Agent(
                name=f"agent_{i}",
                shape=Sphere(radius),
                u_multiplier=max_u,
                mass=mass,
                render_action=True,
            )
            capabilities.append([max_u, agent.shape.radius, agent.mass])
            world.add_agent(agent)
        self.capabilities = torch.tensor(capabilities)
        print(self.capabilities)

        goal = Landmark(
            name="goal",
            collide=False,
            shape=Sphere(),
            color=Color.LIGHT_GREEN,
        )
        world.add_landmark(goal)
        self.package = Landmark(
            name="package",
            collide=True,
            movable=True,
            shape=Sphere(),
            mass=self.package_mass,
            color=Color.RED,
        )
        self.package.goal = goal
        world.add_landmark(self.package)
        # Add landmarks

        self.line = Landmark(
            name="line",
            shape=Line(length=self.line_length),
            collide=True,
            movable=True,
            rotatable=True,
            mass=5,
            color=Color.BLACK,
        )
        world.add_landmark(self.line)

        self.floor = Landmark(
            name="floor",
            collide=True,
            shape=Box(length=10, width=1),
            color=Color.WHITE,
        )
        world.add_landmark(self.floor)

        self.pos_rew = torch.zeros(batch_dim, device=device, dtype=torch.float32)
        self.ground_rew = self.pos_rew.clone()

        if self.eval_seed:
            self.set_rng_state(rng_state, device)

        return world

    def reset_world_at(self, env_index: int = None):
        rng_state = None
        if self.eval_seed:
            rng_state = self.get_rng_state(self.world.device)
            self.set_eval_seed(self.eval_seed)
        
        # reset capabilities, only do this during batched resets!
        if not env_index:        
            capabilities = [] # save capabilities for relative capabilities later
            for agent in self.world.agents:
                if self.multiple_ranges:
                    cap_idx = int(random.choice(np.arange(len(self.capability_mult_range))))
                    self.capability_mult_min = self.capability_mult_range[cap_idx][0]
                    self.capability_mult_max = self.capability_mult_range[cap_idx][1]
                max_u = self.default_u_multiplier * random.uniform(self.capability_mult_min, self.capability_mult_max)
                if self.multiple_ranges:
                    cap_idx = int(random.choice(np.arange(len(self.capability_mult_range))))
                    self.capability_mult_min = self.capability_mult_range[cap_idx][0]
                    self.capability_mult_max = self.capability_mult_range[cap_idx][1]
                radius = self.default_agent_radius * random.uniform(self.capability_mult_min, self.capability_mult_max)
                if self.multiple_ranges:
                    cap_idx = int(random.choice(np.arange(len(self.capability_mult_range))))
                    self.capability_mult_min = self.capability_mult_range[cap_idx][0]
                    self.capability_mult_max = self.capability_mult_range[cap_idx][1]
                mass = self.default_agent_mass * random.uniform(self.capability_mult_min, self.capability_mult_max)

                # capabilities.append([max_u, agent.shape.radius, agent.mass])
                capabilities.append([max_u, radius, mass])

                agent.u_multiplier=max_u
                agent.shape=Sphere(radius)
                agent.mass=mass

            self.capabilities = torch.tensor(capabilities)

        goal_pos = torch.cat(
            [
                torch.zeros(
                    (1, 1) if env_index is not None else (self.world.batch_dim, 1),
                    device=self.world.device,
                    dtype=torch.float32,
                ).uniform_(
                    -1.0,
                    1.0,
                ),
                torch.zeros(
                    (1, 1) if env_index is not None else (self.world.batch_dim, 1),
                    device=self.world.device,
                    dtype=torch.float32,
                ).uniform_(
                    0.0,
                    self.world.y_semidim,
                ),
            ],
            dim=1,
        )
        line_pos = torch.cat(
            [
                torch.zeros(
                    (1, 1) if env_index is not None else (self.world.batch_dim, 1),
                    device=self.world.device,
                    dtype=torch.float32,
                ).uniform_(
                    -1.0 + self.line_length / 2,
                    1.0 - self.line_length / 2,
                ),
                torch.full(
                    (1, 1) if env_index is not None else (self.world.batch_dim, 1),
                    -self.world.y_semidim + self.default_agent_radius * self.capability_mult_max * 2 if not self.multiple_ranges else \
                    -self.world.y_semidim + self.default_agent_radius * self.capability_mult_range[-1][1] * 2,
                    device=self.world.device,
                    dtype=torch.float32,
                ),
            ],
            dim=1,
        )
        package_rel_pos = torch.cat(
            [
                torch.zeros(
                    (1, 1) if env_index is not None else (self.world.batch_dim, 1),
                    device=self.world.device,
                    dtype=torch.float32,
                ).uniform_(
                    -self.line_length / 2 + self.package.shape.radius
                    if self.random_package_pos_on_line
                    else 0.0,
                    self.line_length / 2 - self.package.shape.radius
                    if self.random_package_pos_on_line
                    else 0.0,
                ),
                torch.full(
                    (1, 1) if env_index is not None else (self.world.batch_dim, 1),
                    self.package.shape.radius,
                    device=self.world.device,
                    dtype=torch.float32,
                ),
            ],
            dim=1,
        )

        for i, agent in enumerate(self.world.agents):
            agent.set_pos(
                line_pos
                + torch.tensor(
                    [
                        -(self.line_length - agent.shape.radius) / 2
                        + i
                        * (self.line_length - agent.shape.radius)
                        / (self.n_agents - 1),
                        -agent.shape.radius * 2,
                    ],
                    device=self.world.device,
                    dtype=torch.float32,
                ),
                batch_index=env_index,
            )

        self.line.set_pos(
            line_pos,
            batch_index=env_index,
        )
        self.package.goal.set_pos(
            goal_pos,
            batch_index=env_index,
        )
        self.line.set_rot(
            torch.zeros(1, device=self.world.device, dtype=torch.float32),
            batch_index=env_index,
        )
        self.package.set_pos(
            line_pos + package_rel_pos,
            batch_index=env_index,
        )

        self.floor.set_pos(
            torch.tensor(
                [
                    0,
                    -self.world.y_semidim
                    - self.floor.shape.width / 2
                    - (
                        self.default_agent_radius * self.capability_mult_max if not self.multiple_ranges else \
                        self.default_agent_radius * self.capability_mult_range[-1][1]
                    ),
                ],
                device=self.world.device,
            ),
            batch_index=env_index,
        )
        self.compute_on_the_ground()
        if env_index is None:
            self.global_shaping = (
                torch.linalg.vector_norm(
                    self.package.state.pos - self.package.goal.state.pos, dim=1
                )
                * self.shaping_factor
            )
        else:
            self.global_shaping[env_index] = (
                torch.linalg.vector_norm(
                    self.package.state.pos[env_index]
                    - self.package.goal.state.pos[env_index]
                )
                * self.shaping_factor
            )

        if self.eval_seed:
            self.set_rng_state(rng_state, self.world.device)

    def compute_on_the_ground(self):
        self.on_the_ground = self.world.is_overlapping(
            self.line, self.floor
        ) + self.world.is_overlapping(self.package, self.floor)

    def reward(self, agent: Agent):
        is_first = agent == self.world.agents[0]

        if is_first:
            self.pos_rew[:] = 0
            self.ground_rew[:] = 0

            self.compute_on_the_ground()
            self.package_dist = torch.linalg.vector_norm(
                self.package.state.pos - self.package.goal.state.pos, dim=1
            )

            self.ground_rew[self.on_the_ground] = self.fall_reward

            global_shaping = self.package_dist * self.shaping_factor
            self.pos_rew = self.global_shaping - global_shaping
            self.global_shaping = global_shaping

        return self.ground_rew + self.pos_rew

    def get_capability_repr(self, agent: Agent):
        """
        Get capability representation:
            raw = raw multiplier values
            relative = zero-meaned (taking mean of team into account)
            mixed = raw + relative (concatenated)
        """
        # agent's normal capabilities
        max_u = agent.u_multiplier
        radius = agent.shape.radius
        mass = agent.mass

        # compute the mean capabilities across the team's agents
        # then compute "relative capability" of this agent by subtracting the mean
        team_mean = list(torch.mean(self.capabilities, dim=0))
        rel_max_u = max_u - team_mean[0].item()
        rel_radius = radius - team_mean[1].item()
        rel_mass = mass - team_mean[2].item()

        raw_capability_repr = [
            torch.tensor(
                max_u, device=self.world.device
            ).repeat(self.world.batch_dim, 1),
            torch.tensor(
                radius, device=self.world.device
            ).repeat(self.world.batch_dim, 1),
            torch.tensor(
                mass, device=self.world.device
            ).repeat(self.world.batch_dim, 1),
        ]

        rel_capability_repr = [
            torch.tensor(
                rel_max_u, device=self.world.device
            ).repeat(self.world.batch_dim, 1),
            torch.tensor(
                rel_radius, device=self.world.device
            ).repeat(self.world.batch_dim, 1),
            torch.tensor(
                rel_mass, device=self.world.device
            ).repeat(self.world.batch_dim, 1),
        ]

        if self.capability_representation == "raw":
            return raw_capability_repr
        elif self.capability_representation == "relative":
            return rel_capability_repr
        elif self.capability_representation == "mixed":
            return raw_capability_repr + rel_capability_repr

    def observation(self, agent: Agent):
        # get positions of all entities in this agent's reference frame
        capability_repr = self.get_capability_repr(agent)
        return torch.cat(
            [
                agent.state.pos,
                agent.state.vel,
                agent.state.pos - self.package.state.pos,
                agent.state.pos - self.line.state.pos,
                self.package.state.pos - self.package.goal.state.pos,
                self.package.state.vel,
                self.line.state.vel,
                self.line.state.ang_vel,
                self.line.state.rot % torch.pi,
            ] + capability_repr,
            dim=-1,
        )

    def done(self):
        return self.on_the_ground + self.world.is_overlapping(
            self.package, self.package.goal
        )

    def info(self, agent: Agent):
        self.success_rate = self.world.is_overlapping(
            self.package, self.package.goal
        )
        info = {"pos_rew": self.pos_rew, "ground_rew": self.ground_rew, "success_rate": self.success_rate}
        return info


class HeuristicPolicy(BaseHeuristicPolicy):
    def compute_action(self, observation: torch.Tensor, u_range: float) -> torch.Tensor:
        batch_dim = observation.shape[0]

        index_package_goal_pos = 8
        dist_package_goal = observation[
            :, index_package_goal_pos : index_package_goal_pos + 2
        ]
        y_distance_ge_0 = dist_package_goal[:, Y] >= 0

        if self.continuous_actions:
            action_agent = torch.clamp(
                torch.stack(
                    [
                        torch.zeros(batch_dim, device=observation.device),
                        -dist_package_goal[:, Y],
                    ],
                    dim=1,
                ),
                min=-u_range,
                max=u_range,
            )
            action_agent[:, Y][y_distance_ge_0] = 0
        else:
            action_agent = torch.full((batch_dim,), 4, device=observation.device)
            action_agent[y_distance_ge_0] = 0
        return action_agent


if __name__ == "__main__":
    render_interactively(
        __file__,
        n_agents=3,
        package_mass=5,
        random_package_pos_on_line=True,
        control_two_agents=True,
    )
