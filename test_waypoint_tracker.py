from vmas import make_env
import numpy as np
import torch

from vmas.simulator.utils import save_video


DEVICE="cuda"
SEED=1

env = make_env(
    scenario="navigation",
    num_envs=1,
    max_steps=200,
    device=DEVICE,
    continuous_actions=True,
    wrapper=None,
    seed=SEED,
    # Environment specific variables
    n_agents=1,
)

obs = env.reset()
frame_list = []
for i in range(100):
    if i == 0:
        agent = env.agents[0]
        agent.goal.state.pos = torch.tensor([[0.0, 1.0]])
        agent.state.pos = torch.tensor([[0.0, 0.0]])
        # agent.state.rot = torch.tensor([[0.0]]) # this doesn't work

    # print(i)
    # act once for each agent
    act = []
    for i, agent in enumerate(env.agents):

        # find goal w.r.t robot frame (difference given in global frame)
        # robot frame = FLU
        rel_pose_to_goal = agent.goal.state.pos - agent.state.pos
        goal_x_global = rel_pose_to_goal[:, 0]
        goal_y_global = rel_pose_to_goal[:, 1]
        agent_x_global = agent.state.pos[:, 0]
        agent_y_global = agent.state.pos[:, 1]
        theta_robot_to_global = -agent.state.rot

        # print(f'({agent_x_global.item()}, {agent_y_global.item()})')
        # print("agent state")
        # print(agent_x_global, agent_y_global, theta_robot_to_global)
        # print("rel goal global")
        # print(goal_x_global, goal_y_global)
        
        goal_x_robot = goal_x_global * torch.cos(theta_robot_to_global) - goal_y_global * torch.sin(theta_robot_to_global)
        goal_y_robot = goal_x_global * torch.sin(theta_robot_to_global) + goal_y_global * torch.cos(theta_robot_to_global)

        # print("goal_robot_coords", goal_x_robot, goal_y_robot)

        to_goal = torch.cat([goal_x_robot, goal_y_robot], dim=1)
        # print("to goal", to_goal)

        action = to_goal
        action = (to_goal / torch.linalg.norm(to_goal)) * 1e-1
        # action = torch.tensor([[0.00, 0.01]])
        print("raw action input", action)
        act.append(action)

    next, rew, done, info = env.step(act)

    frame = env.render(
        mode="rgb_array",
        visualize_when_rgb=True,
    )
    frame_list.append(frame)

save_video(
    "test_waypoint_tracker",
    frame_list,
    fps=10,
)
