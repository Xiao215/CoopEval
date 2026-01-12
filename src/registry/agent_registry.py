from src.agents.agent_manager import Agent, CoTAgent, IOAgent
import copy

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


def create_players_with_player_id(
    agent_cfgs: list[dict], num_players: int
) -> list[Agent]:
    """Create players with fixed player IDs from agent configurations."""
    players = []
    for cfg in agent_cfgs:
        for player_id in range(1, num_players + 1):
            agent_config = copy.deepcopy(cfg)
            agent_config["player_id"] = player_id
            agent = create_agent(agent_config)
            players.append(agent)
    return players
