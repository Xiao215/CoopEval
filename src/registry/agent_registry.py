from src.agents.agent_manager import Agent, CoTAgent, IOAgent

AGENT_REGISTRY = {
    "IOAgent": IOAgent,
    "CoTAgent": CoTAgent,
}


def create_agent(agent_config: dict) -> Agent:
    agent_class = AGENT_REGISTRY.get(agent_config["type"])

    if agent_class is None:
        raise ValueError(f"Unknown agent type: {agent_config['type']}")

    agent = agent_class(agent_config=agent_config)

    return agent
