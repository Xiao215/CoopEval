from src.agents.agent_manager import Agent, CoTAgent, IOAgent

AGENT_REGISTRY = {
    "IOAgent": IOAgent,
    "CoTAgent": CoTAgent,
}


def create_agent(agent_config: dict, player_id: int) -> Agent:
    agent_class = AGENT_REGISTRY.get(agent_config["type"])

    if agent_class is None:
        raise ValueError(f"Unknown agent type: {agent_config['type']}")

    agent_config["player_id"] = player_id
    agent = agent_class(agent_config=agent_config)

    return agent
