"""Central repository for prompt templates used by mechanisms."""

import textwrap

# Contracting mechanism prompts
CONTRACT_DESIGN_PROMPT = textwrap.dedent(
    """
    Here is the twist:
    There will be the option for a payment contract in this game, and your task now is to design and propose one.

    - A contract is an additional payoff agreement on top of the original game payoffs. It specifies a number for each action that a player can play, indicating one of three cases:
        * Positive number (+): the player receives an additional payment of X points in total, drawn equally from the other players.
        * Negative number (-): the player pays an additional payment of X points in total, distributed equally among the other players.
        * Zero (0): no additional payments in either direction.
    - Each player may choose to accept the contract as a whole or not.
    - The contract becomes active only if all players accept.

    The other players will also design and propose a contract. Only one will be present in the game though. Which one will be decided in a separate step later via an approval voting process by you and the other players. The winning contract will be selected uniform at random from those with the maximum number of approvals.

    Output Format:
    Return a valid JSON object in a single line:
    {{"A0": <INT>, "A1": <INT>, ...}}

    - Keys: all available game actions.
    - Values: integers representing the extra payoff for that action.
    """
)

CONTRACT_APPROVAL_VOTE_PROMPT = textwrap.dedent(
    """
    Here is the twist:
    On top of the original game rules, a payment contract can be put in place if players agree to it via an approval voting process. A contract specifies a payment value for each action that a player can play.

    Your task now is to review each proposed contract and decide which ones you approve of. The winning contract will be selected uniform at random from those with the maximum number of approvals.

    Here are the contract designs that have been proposed:
    {all_contracts_description}

    Output Format:
    Return a valid JSON object with your approvals:
    {{"C1": <true/false>, "C2": <true/false>, ...}}

    - Keys: contract identifiers (e.g., "C1", "C2", ...)
    - Values: `true` if you approve, `false` if you don't
    - Ensure all contracts have an entry
    """
)

CONTRACT_CONFIRMATION_PROMPT = textwrap.dedent(
    """
    Here is the twist:
    On top of the original game rules, you have the option to sign a payment contract. A contract specifies a payment value for each action that a player can play. Here is the contract that was selected via approval voting (proposed by Player {designer_player_id}):
    {contract_description}

    At this stage, you are asked to decide whether to sign the contract. The contract becomes active only if all players sign it.

    Output Requirement:
    - Respond with a valid JSON object.
    - Format: {{"sign": <BOOL>}} where <BOOL> is true or false.
    """
)

CONTRACT_MECHANISM_PROMPT = textwrap.dedent(
    """
    Here is the twist:
    On top of the original game rules, there is a payment contract in place because every player signed it in beforehand. Here is the contract that was selected via approval voting (proposed by Player {designer_player_id}):
    {contract_description}

    Since this contract directly affects your final payoff, consider the contract when making your strategy decisions!
    """
)

CONTRACT_REJECTION_PROMPT = textwrap.dedent(
    """
    Here is the twist:
    On top of the original game rules, a payment contract was proposed to the players. It is NOT active though because some players rejected it. Here is the proposed contract that was selected via approval voting (proposed by Player {designer_player_id}):
    {contract_description}

    Players who rejected it: {rejector_ids}

    You will now play the original game without any contract modifications.
    """
)

# Disarmament mechanism prompts
DISARM_PROMPT_BASE = textwrap.dedent(
    """
    Here is the twist:
    You and the other players are currently in a disarmament phase, where over multiple rounds, each of you have the option to "disarm" actions in advance. You can do that for a particular action by setting an "upper bound" commitment (in %) to the maximum probability with which you may decide to take that action in the original game.

    Your current upper bounds:
        {my_caps}
    Other players' current upper bounds:
        {other_players_caps}

    Rules:
    1) For each action, you may keep the upper bound the same or reduce it. Increases are forbidden.
    2) Each upper bound must be an integer in [0, 100].
    3) All upper bounds must be non-negative and the sum of your upper bounds must be greater than or equal to 100.
    4) All players are facing the question of whether to disarm simultaneously, and players' decisions to disarm are and will be reflected in the current upper bounds (already reported above).
    5) Each round, you must make one of three choices:
       - "disarm": Strictly reduce at least one upper bound (you must then provide new bounds that differ from current for at least one action).
       - "pass": Skip this round but remain willing to wait for others to disarm.
       - "end": Veto the remaining disarmament phase and stop it for everyone.
    6) Continuation rules:
       - If ANY player chooses "end", the disarmament phase stops immediately and ANY disarming occurring in that round will not be applied.
       - If NO player chooses "disarm" (for example, everyone chooses "pass"), the disarmament phase stops.
       - If at least one player chooses "disarm" and no one chooses "end", there is a {discount}% chance probability that an additional round will take place.
    7) After the disarmament phase ends, you and the other players will play the original game subject to your committed probability upper bound constraints.
    """
)

DISARM_FORMAT_CAN_DISARM = textwrap.dedent(
    """
    Format requirement:
    Return your choice and (if you choose to disarm) new probability upper bounds as a JSON object:
    {{"choice": "disarm", "A0": <INT>, "A1": <INT>, ...}}
    OR
    {{"choice": "pass"}}
    OR
    {{"choice": "end"}}

    - "choice" must be one of: "disarm", "pass", or "end"
    - If you choose "disarm", you MUST include all action keys (A0, A1, ...) with integer values, and at least one cap must be lower than your current bounds
    - If you choose "pass" or "end", do NOT include action keys
    """
)

DISARM_FORMAT_CANNOT_DISARM = textwrap.dedent(
    """
    NOTE: Your upper bounds already sum to 100, so you have no further room to disarm.

    Format requirement:
    Return your choice as a JSON object:
    {{"choice": "pass"}}
    OR
    {{"choice": "end"}}

    - "choice" must be one of: "pass" or "end"
    """
)

DISARMAMENT_MECHANISM_PROMPT = textwrap.dedent(
    """
    Here is the twist:
    There was a disarmament phase between you and the other players, in which each of you had the option to "disarm" actions in advance. This was done for a particular action by setting an "upper bound" commitment (in %) to the maximum probability with which you may now decide to take that action in the game. The following upper bounds arose from that disarmament phase:

    Your upper bounds:
        {my_caps}
    Other players' upper bounds:
        {other_players_caps}

    The disarmament phase ended for the following reason: {termination_reason}

    In addition to the instructions below, you must now propose a probability distribution over actions subject to your committed probability upper bound constraints.
    """
)

# Mediation mechanism prompts
MEDIATOR_DESIGN_PROMPT = textwrap.dedent(
    """
    Here is the twist:
    There will be a mediator for this game, and your task now is to design and propose one.

    - A mediator is an entity that plays actions on behalf of delegating players.
    - Each player may choose to delegate their move to the mediator or act independently.
    - The mediator observes the number of players delegating to the mediator and then plays the same action for all delegating players.

    The other players will also design and propose a mediator. Only one will be present in the game though. Which one will be decided in a separate step later via an approval voting process by you and the other players. The winning mediator will be selected uniform at random from those with the maximum number of approvals.

    Output Format:
    Return a valid JSON object in a single line:
    {{"1": <Action>, ..., "{num_players}": <Action>}} where <Action> is a string like "A0", "A1" ...

    - Keys: the number of players delegating (from 1 to {num_players}).
    - Values: the action the mediator will play on behalf of delegating players (e.g., "A0" or "A1" etc.).
    """
)

MEDIATOR_APPROVAL_VOTE_PROMPT = textwrap.dedent(
    """
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
    """
)

MEDIATION_MECHANISM_PROMPT = textwrap.dedent(
    """
    Here is the twist:
    On top of the original game rules, you have the option to delegate your move to a mediator.
    If you choose to delegate, the mediator will play an action for you based on how many players have delegated to it.
    You can also choose to act independently.

    The available mediator was proposed by Player {designer_player_id} and selected via approval voting among the players. Here is what the mediator would do for the players that delegate to it:
    {mediator_description}

    Consider A{additional_action_id} as an additional action "Delegate to Mediator". Your final mixed strategy should include probability for all actions A0, A1, ..., A{additional_action_id}.
    """
)


# Repetition mechanism prompts
REPETITION_MECHANISM_PROMPT = textwrap.dedent(
    """
    Here is the twist:
    You are playing this game *repeatedly* with the same player(s). The action sampled from your action probability distribution will be visible to those players in future rounds and may influence their decisions.
    You are currently playing round {round_idx} of the game.
    After each round, there is a {discount}% chance probability that an additional round will take place.

    Next, you find the info available to you about past few rounds of interaction:

    {history_context}
    """
)
REPETITION_NO_HISTORY_DESCRIPTION = (
    "You haven't played any rounds with the other player(s) yet."
)
REPETITION_ROUND_LINE = "[Round {round_idx}] \n{actions}"
REPETITION_SELF_LABEL = "\tYou: {action}"
REPETITION_OTHERPLAYER_LABEL = "\t{other_player}: {action}"

# Reputation mechanism prompts
REPUTATION_MECHANISM_PROMPT = textwrap.dedent(
    """
    Here is the twist:
    You are playing this game *multiple times* with many different players randomly.
    The action sampled from your action probability distribution will be visible to everyone in future rounds and may influence their decisions.
    You are currently playing round {round_idx} of the game.
    After each round, there is a {discount}% chance probability that an additional round will take place.

    Next, you find the info available to you about the reputation information regarding your current opponent(s)

    {history_context}
    """
)

REPUTATION_NO_HISTORY_DESCRIPTION = (
    "{opponent_name} has no prior history of playing this game."
)
REPUTATION_NO_ACTION_DISTRIBUTION_DESCRIPTION = (
    "{opponent_name} has no prior action distribution"
)
