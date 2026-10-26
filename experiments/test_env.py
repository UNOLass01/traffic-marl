import sumo_rl

env = sumo_rl.parallel_env(
    net_file="external/sumo-rl/sumo_rl/nets/4x4-Lucas/4x4.net.xml",
    route_file="external/sumo-rl/sumo_rl/nets/4x4-Lucas/4x4c1c2c1c2.rou.xml",
    use_gui=False,
    num_seconds=500,
)

observations, infos = env.reset()
print("Agents (traffic lights):", env.agents)

while env.agents:
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}
    observations, rewards, terms, truncs, infos = env.step(actions)

env.close()
print("Simulation finished successfully")
