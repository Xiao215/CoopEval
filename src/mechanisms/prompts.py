"""Central repository for prompt templates used by mechanisms."""

import textwrap

# Contracting mechanism prompts
CONTRACT_DESIGN_PROMPT = textwrap.dedent(
    """
    Here is the twist:
    There will be the option for a payment contract in this game, and your task now is to design and propose one.

    - A contract is an additional payoff agreement on top of the original game payoffs. It specifies a number for each action that a player can play, indicating one of three cases:
        * Positive number (+): the player receives an additional payment of X points in total, drawn equally from the other players.
        * Negative number (-): the player pays an additional payment of Y points in total, distributed equally among the other players.
        * Zero (0): no additional payments in either direction.    
    - Each player may choose to accept the contract as a whole or not.
    - The contract becomes active only if all players accept.

    The other players will also design and propose a contract. Only one will be present in the game though. Which one this is will be decided in a separate step later via an approval voting process by you and the other players. The winning contract will be selected uniform at random from those with the maximum number of approvals.

    Output Format:
    Return a valid JSON object in a single line:
    {{"A0": <INT>, "A1": <INT>, ...}}

    - Keys: all available game actions.
    - Values: integers representing the extra payoff for that action.
    """
)

CONTRACT_CONFIRMATION_PROMPT = textwrap.dedent(
    """
    Here is the twist:
    On top of the original game rules, you have the option to sign a payment contract. The contract specifies a number for each action that a player can play, indicating one of three cases:
        * Positive number (+): the player receives an additional payment of X points in total, drawn equally from the other players.
        * Negative number (-): the player pays an additional payment of Y points in total, distributed equally among the other players.
        * Zero (0): no additional payments in either direction.  

    Here is the contract:
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
    On top of the original game rules, there is a payment contract in place because every player signed it in beforehand. Here is the contract:
    {contract_description}

    Since this contract directly affects your final payoff, consider the contract when making your strategy decisions!
    """
)

CONTRACT_REJECTION_PROMPT = textwrap.dedent(
    """
    Here is the twist:
    On top of the original game rules, a payment contract was proposed to the players. It is NOT activate though because some players rejected it. Here is the proposed contract:
    {contract_description}

    The number of players who rejected it is {num_rejectors}.

    You will now play the original game without any contract modifications.
    """
)

# Disarmament mechanism prompts
DISARM_PROMPT = textwrap.dedent(
    """
    Instruction:
    You are negotiating a multi-action disarmament for this game.

    A "cap" is the maximum probability (in %) with which you may choose an action in this game.
    Your current caps:
        {my_caps}
    Opponents' current caps:
        {opponents_caps}

    Rules:
    1) For each action, you may keep the cap the same or reduce it. Increases are forbidden.
    2) Each cap must be an integer in [0, 100].
    3) All caps must be non-negative and the sum of your caps must be greater than or equal to 100.
    4) Moves are simultaneous; assume others facing the same negotiation.
    5) If at least one party reduces any cap, negotiations continue to another round with probability {discount}% (otherwise they end).
    6) If every party leaves all caps unchanged, negotiations end immediately.


    Format requirement:
    Return the new cap as a JSON object, for example:
    {{"A0": <INT>, "A1": <INT>, ...}}
    """
)

DISARMAMENT_MECHANISM_PROMPT = textwrap.dedent(
    """
    Additional Information:
    A "cap" is the maximum probability (in %) with which you may choose an action in the next round.
    From previous round of negotiation, you agree to have a cap of:
    {caps_str}

    Now you need to propose a new probability distribution over actions subjected to your current cap limits.
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

    The other players will also design and propose a mediator. Only one will be present in the game though. Which one this is will be decided in a separate step later via an approval voting process by you and the other players. The winning mediator will be selected uniform at random from those with the maximum number of approvals.

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

    Here is what the mediator would do for the players that delegate to it:
    {mediator_description}

    Consider A{additional_action_id} as an additional action "Delegate to Mediator". Your final mixed strategy should include probability for all actions A0, A1, ..., A{additional_action_id}.
    """
)


# Repetition mechanism prompts
REPETITION_MECHANISM_PROMPT = textwrap.dedent(
    """
    Here is the twist:
    You are playing this game repeatedly with the same players. The action sampled from your action probability distribution will be visible to those players in future rounds and may influence their decisions.
    You are currently playing round {round_idx} of the game.
    History:
    {history_context}
    """
)
REPETITION_NO_HISTORY_DESCRIPTION = (
    "You haven't played any rounds with these opponent(s) yet."
)
REPETITION_ROUND_LINE = "[Round {round_idx}] \n{actions}"
REPETITION_RECENT_ROUND_LINE = "[{relative_idx} round(s) ago] \n{actions}"
REPETITION_RECENT_OPPONENT_DIST_HEADER = (
    "Opponents' action counts over last {window} round(s):"
)
REPETITION_RECENT_HISTORY_PROMPT = textwrap.dedent(
    """
    History over the last {window_count} round(s):
    {recent_history}
    """
)
REPETITION_SELF_LABEL = "\tYou: {action}"
REPETITION_OPPONENT_LABEL = "\t{opponent}: {action}"
