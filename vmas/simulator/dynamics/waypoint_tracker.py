"""
Implements a waypoint tracker, assuming a DifferentialDrive robot.

Waypoints are given as relative position and heading, or u=(dx, dy, dtheta).
"""

import math

import torch

import vmas.simulator.core
import vmas.simulator.utils
from vmas.simulator.dynamics.common import Dynamics
from vmas.simulator.dynamics.diff_drive import DiffDrive


class WaypointTracker(Dynamics):
    def __init__(
        self,
        world: vmas.simulator.core.World,
        speed=1e-1,
    ):
        super().__init__()

        self.dt = world.dt
        self.world = world
        self.speed = speed

    @property
    def needed_action_size(self) -> int:
        return 2

    def process_action(self):
        # TODO: add input queue, see VelocityController

        # convert relative position (dx, dy) to (forward velocity, rotational velocity)
        # TODO: verify FLU
        dx = self.agent.action.u[:, 0]
        dy = self.agent.action.u[:, 1]

        # round dx/dy to zero if near 0
        dx = torch.where(torch.abs(dx) < 1e-3, 0, dx)
        dy = torch.where(torch.abs(dy) < 1e-3, 0, dy)

        # when x is near 0, set angular_velocity to either -pi/2 or pi/2, depending on sign of y
        # otherwise, set it to clamp(arctan(dy/dx) / dt), clamped to +/- pi
        angular_velocity = torch.where(torch.abs(dx) < 1e-2, 
                                       torch.sign(dy) * torch.pi/2, 
                                       torch.clamp(torch.arctan(dy/dx) / self.dt, -torch.pi, torch.pi))

        forward_velocity = dx / self.dt
        # print("vel inputs", forward_velocity, angular_velocity)

        robot_angle = self.agent.state.rot
        linear_accel = torch.cat([forward_velocity * torch.cos(robot_angle), forward_velocity * torch.sin(robot_angle)], dim=0)  / self.dt
        angular_accel = angular_velocity / self.dt
        # print("accel", linear_accel, angular_accel)

        # print(self.agent.mass, self.agent.moment_of_inertia)
        linear_force = self.agent.mass * linear_accel
        angular_force = self.agent.moment_of_inertia * angular_accel
        print("forces", linear_force, angular_force)

        force_factor = self.speed
        self.agent.state.force[:, 0] = linear_force[0] * force_factor
        self.agent.state.force[:, 1] = linear_force[1] * force_factor
        self.agent.state.torque = angular_force.unsqueeze(-1) * force_factor




