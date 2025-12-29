"""Central repository for prompt templates used by mechanisms."""

import textwrap

# Contracting mechanism prompts
CONTRACT_DESIGN_PROMPT = textwrap.dedent(
    """
    Instruction:
    You are tasked with designing a **contract** for this game.
    - A contract is an extra payoff agreement.
    - Each player may choose to sign it or not.
    - The contract is valid only if all players sign.
    - For each action:
        * Positive number (+): the player receives X payoff from others.
        * Negative number (-): the player pays Y payoff to others.
        * Zero (0): no extra payoff.
    - Goal: maximize the total payoff for all players if the contract is signed.

    Output Requirement:
    - Return exactly **one valid Python dictionary** on a single line.
    - Format: {"A0": <INT>, "A1": <INT>, ...}
    - Keys: all available game actions.
    - Values: integers representing the extra payoff for that action.
    - Ensure the dictionary is syntactically valid Python.
    """
)

CONTRACT_CONFIRMATION_PROMPT = textwrap.dedent(
    """
    Contract Rule:
    On top of the original game instructions, you have the option to sign a contract.
    A contract is an extra payoff agreement that is valid only if all players sign.
    Here is the contract:
    {contract_description}

    Output Requirement:
    - Respond with a valid JSON object.
    - Format: {"sign": <BOOL>} where <BOOL> is true or false.
    """
)

CONTRACT_MECHANISM_PROMPT = textwrap.dedent(
    """
    Contract Rule:
    On top of the original game instructions, everyone has agreed to sign a contract.
    Here is the contract:
    {contract_description}

    Since this contract directly change your final payoff,
    consider the contract when making your decision for the strategy!
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
    Instruction:
    You are tasked with designing a mediator agent for this game.

    - A mediator is an agent that suggests actions to players.
    - Each player may choose to delegate their move to the mediator or act independently.
    - The mediator observes number of players delegating to the mediator and then recommends the same action to all delegating players.
    - The mediator's objective is to maximize the payoff of the players who delegate to the mediator.

    Output Format:
    Return a valid json in a single line:
    {{"1": <Action>, ..., "{num_players}": <Action>}} where <Action> is a string like "A0", "A1" ...

    - Keys: the number of players delegating (from 1 to {num_players}).
    - Values: the action the mediator will recommend (e.g., "A0", "A1", ...).
    - Ensure the dictionary is syntactically valid in Python.
    """
)

MEDIATION_MECHANISM_PROMPT = textwrap.dedent(
    """
    Additional Information:
    On top of the original game instructions, you have the option to delegate your move to a mediator agent.
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
    This is a repeated game, so your chosen action will be visible to the same opponent(s) in future rounds and may influence their decisions.
    You are currently playing round {round_idx} of the game.
    History:
    {history_context}
    """
)
REPETITION_NO_HISTORY_DESCRIPTION = (
    "You haven't played any rounds with these opponent(s) yet."
)
REPETITION_ROUND_LINE = "[Round {round_idx}] \n{actions}"
REPETITION_RECENT_ROUND_LINE = "[Last {relative_idx} round] \n{actions}"
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
