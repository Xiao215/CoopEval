"""Central repository for prompt templates used by mechanisms."""

import textwrap

# Contracting mechanism prompts
CONTRACT_DESIGN_PROMPT = textwrap.dedent("""
    Here is the twist:
    There will be the option for a payment contract in this game, and your task now is to design and propose one.

    - A contract is an additional payoff agreement on top of the original game payoffs. It specifies a number for each action that a player can play, indicating one of three cases:
        * Positive number (+): the player receives an additional payment of X points in total, drawn equally from the other player(s).
        * Negative number (-): the player pays an additional payment of X points in total, distributed equally among the other player(s).
        * Zero (0): no additional payments in either direction.
    - Each player may choose to accept the contract as a whole or not.
    - The contract becomes active only if all players accept.

    The other player(s) will also design and propose a contract. Only one will be present in the game though. Which one will be decided in a separate step later via an approval voting process by you and the other player(s). The winning contract will be selected uniform at random from those with the maximum number of approvals.

    Output Format:
    Return a valid JSON object in a single line:
    {{"A0": <INT>, "A1": <INT>, ...}}

    - Keys: all available game actions.
    - Values: integers representing the extra payoff for that action.
    """)

CONTRACT_APPROVAL_VOTE_PROMPT = textwrap.dedent("""
    Here is the twist:
    On top of the original game rules, a payment contract can be put in place if the players agree to it via an approval voting process. A contract specifies a payment value for each action that a player can play.

    Your task now is to review each proposed contract and decide which ones you approve of. The winning contract will be selected uniform at random from those with the maximum number of approvals.

    Here are the contract designs that have been proposed:
    {all_contracts_description}

    Output Format:
    Return a valid JSON object with your approvals:
    {{"C1": <true/false>, "C2": <true/false>, ...}}

    - Keys: contract identifiers (e.g., "C1", "C2", ...)
    - Values: `true` if you approve, `false` if you don't
    - Ensure all contracts have an entry
    """)

CONTRACT_CONFIRMATION_PROMPT = textwrap.dedent("""
    Here is the twist:
    On top of the original game rules, you have the option to sign a payment contract. A contract specifies a payment value for each action that a player can play. Here is the contract that was selected via approval voting (proposed by Player {designer_player_id}):
    {contract_description}

    At this stage, you are asked to decide whether to sign the contract. The contract becomes active only if all players sign it.

    Output Requirement:
    - Respond with a valid JSON object.
    - Format: {{"sign": <BOOL>}} where <BOOL> is true or false.
    """)

CONTRACT_MECHANISM_PROMPT = textwrap.dedent("""
    Here is the twist:
    On top of the original game rules, there is a payment contract in place because every player signed it in beforehand. Here is the contract that was selected via approval voting (proposed by Player {designer_player_id}):
    {contract_description}

    Since this contract directly affects your final payoff, consider the contract when making your strategy decisions!
    """)

CONTRACT_REJECTION_PROMPT = textwrap.dedent("""
    Here is the twist:
    On top of the original game rules, a payment contract was proposed to the players. It is NOT active though because some players rejected it. Here is the proposed contract that was selected via approval voting (proposed by Player {designer_player_id}):
    {contract_description}

    Player(s) who rejected it: {rejector_ids}

    You will now play the original game without any contract modifications.
    """)

# Mediation mechanism prompts
MEDIATOR_DESIGN_PROMPT = textwrap.dedent("""
    Here is the twist:
    There will be a mediator for this game, and your task now is to design and propose one.

    - A mediator is an entity that plays actions on behalf of delegating players.
    - Each player may choose to delegate their move to the mediator or act independently.
    - The mediator observes the number of players delegating to the mediator and then plays the same action for all delegating players.

    The other player(s) will also design and propose a mediator. Only one will be present in the game though. Which one will be decided in a separate step later via an approval voting process by you and the other player(s). The winning mediator will be selected uniform at random from those with the maximum number of approvals.

    Output Format:
    Return a valid JSON object in a single line:
    {{"1": <Action>, ..., "{num_players}": <Action>}} where <Action> is a string like "A0", "A1" ...

    - Keys: the number of players delegating (from 1 to {num_players}).
    - Values: the action the mediator will play on behalf of delegating players (e.g., "A0" or "A1" etc.).
    """)

MEDIATOR_APPROVAL_VOTE_PROMPT = textwrap.dedent("""
    Here is the twist:
    On top of the original game rules, you will have the option to delegate your move to a mediator.
    If you choose to delegate, the mediator will play an action for you based on how many players have delegated to it.
    You can also choose to act independently.

    But first, you and the other player have to decide via an approval voting process which mediator will be present in the game. Your task now is to review each mediator and decide which ones you approve of. The winning mediator will be selected uniform at random from those with the maximum number of approvals.

    Here are the mediator designs that have been proposed:
    {all_mediators_description}

    Output Format:
    Return a valid JSON object with your approvals:
    {{"M1": <true/false>, "M2": <true/false>, ...}}

    - Keys: mediator identifiers (e.g., "M1", "M2", ...)
    - Values: `true` if you approve, `false` if you don't
    - Ensure all mediators have an entry
    """)

MEDIATION_MECHANISM_PROMPT = textwrap.dedent("""
    Here is the twist:
    On top of the original game rules, you have the option to delegate your move to a mediator.
    If you choose to delegate, the mediator will play an action for you based on how many players have delegated to it.
    You can also choose to act independently.

    The available mediator was proposed by Player {designer_player_id} and selected via approval voting among the players. Here is what the mediator would do for the players that delegate to it:
    {mediator_description}

    Consider A{additional_action_id} as an additional action "Delegate to Mediator". Your final mixed strategy should include probability for all actions A0, A1, ..., A{additional_action_id}.
    """)


# Repetition mechanism prompts
REPETITION_MECHANISM_PROMPT = textwrap.dedent("""
    Here is the twist:
    You are playing this game *repeatedly* with the same player(s). The action sampled from your action probability distribution will be visible to those player(s) in future rounds and may influence their decisions.
    After each round, there is a {discount}% chance probability that an additional round will take place. You have already played this game for {round_idx} round(s) in the past.

    Next, you find the info available to you about the history of play that is related to you and the other player(s) you are playing with in this upcoming round.

    {history_context}
    """)
REPETITION_NO_HISTORY_DESCRIPTION = (
    "You haven't played any rounds with the other player(s) yet."
)
REPETITION_ROUND_LINE = "[Round {round_idx}] \n{actions}"
REPETITION_SELF_LABEL = "\tYou: {action_token}"
REPETITION_OTHERPLAYER_LABEL = "\t{other_player}: {action_token}"

# Reputation mechanism prompts
REPUTATION_MECHANISM_PROMPT = textwrap.dedent("""
    Here is the twist:
    You are playing this game *repeatedly* but with varying players who you encounter at random.
    The action sampled from your action probability distribution in the current round will be visible to the players you encounter in future rounds and may influence their decisions.
    After each round, there is a {discount}% chance probability that an additional round will take place. You have already played this game for {round_idx} round(s) in the past.

    Next, you find the info available to you about the history of play that is related to you and the other player(s) you are playing with in this upcoming round.

    {history_context}
    """)

REPUTATION_NO_HISTORY_DESCRIPTION = (
    "{name_plus_have} no prior history of playing this game."
)
REPUTATION_NO_ACTION_DISTRIBUTION_DESCRIPTION = "{indent} → For context, up until round {window_start_minus_one}, {name_plus_have} not played any actions yet."
REPUTATION_PLAYERS_HEADER = (
    "You are playing with {num_opponents} other agent(s): {opponent_uids}."
)
REPUTATION_ACTION_DISTRIBUTION = "{indent} → For context, up until round {window_start_minus_one}, {name_plus_have} played actions as often as follows: {stats_str}"
