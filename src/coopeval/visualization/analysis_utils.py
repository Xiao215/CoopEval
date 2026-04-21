"""Shared utilities for package visualization modules."""

from typing import Any


class NormalizeScore:
    """Normalizes game scores to [0, 1] scale based on Nash Equilibrium and cooperative payoffs.

    Example:
        >>> normalizer = NormalizeScore("PrisonersDilemma", config)
        >>> normalized = normalizer.normalize(raw_score)
    """

    def __init__(self, game: str, game_config: dict):
        """Initialize normalizer by precomputing NE and cooperative payoffs."""
        self.game = game

        if game == "PrisonersDilemma":
            payoff_matrix = game_config["kwargs"]["payoff_matrix"]
            self.ne_payoff = payoff_matrix["DD"][0]  # NE: both defect
            self.coop_payoff = payoff_matrix["CC"][
                1
            ]  # Cooperative: both cooperate

        elif game == "PublicGoods":
            self.coop_payoff = game_config["kwargs"]["multiplier"]
            self.ne_payoff = 1  # NE: no one contributes

        elif game == "TravellersDilemma":
            self.ne_payoff = game_config["kwargs"]["min_claim"]
            spacing = game_config["kwargs"]["claim_spacing"]
            num_actions = game_config["kwargs"]["num_actions"]
            self.coop_payoff = self.ne_payoff + spacing * (num_actions - 1)

        elif game == "TrustGame":
            payoff_matrix = game_config["kwargs"]["payoff_matrix"]
            self.ne_payoff = payoff_matrix["KK"][0]  # NE: both keep
            self.coop_payoff = payoff_matrix["GG"][0]  # Cooperative: both give

        elif game == "StagHunt":
            self.ne_payoff = 3
            self.coop_payoff = 5

        elif game == "MatchingPennies":
            self.ne_payoff = -1
            self.coop_payoff = 0

        else:
            # For non-social-dilemma games, use identity normalization
            self.ne_payoff = 0.0
            self.coop_payoff = 1.0

    def normalize(self, score: float) -> float:
        """Normalize a score to [0, 1] scale (0 = NE payoff, 1 = Cooperative payoff)."""
        if self.coop_payoff == self.ne_payoff:
            # Avoid division by zero
            return 0.0

        return (score - self.ne_payoff) / (self.coop_payoff - self.ne_payoff)

    def denormalize(self, normalized_score: float) -> float:
        """Convert a normalized score back to raw score."""
        return (
            normalized_score * (self.coop_payoff - self.ne_payoff)
            + self.ne_payoff
        )


def is_reputation_mechanism(mechanism_type: str) -> bool:
    """Check if mechanism is any variant of Reputation.

    Handles disambiguated keys of the form "Type (suffix)".
    """
    base = mechanism_type.split(" (")[0]
    return base.lower() in ["reputation", "reputationfirstorder"]


# Validation utilities for multi-folder experiment processing


def validate_folder_count_consistency(
    grouped_data: dict[tuple[str, str], Any],
) -> int:
    """Validate all game+mechanism combinations have same number of folders."""
    if not grouped_data:
        raise ValueError("No groups to validate")

    # Get folder counts for all groups
    folder_counts = {}
    for group_key, group_data in grouped_data.items():
        if isinstance(group_data, dict) and "folders" in group_data:
            # plot_payoff_tensors.py style
            folder_counts[group_key] = len(group_data["folders"])
        elif isinstance(group_data, list):
            # generate_tables.py style
            folder_counts[group_key] = len(group_data)
        else:
            raise ValueError(
                f"Unexpected group_data type for {group_key}: {type(group_data)}"
            )

    unique_counts = set(folder_counts.values())

    if len(unique_counts) != 1:
        # Build detailed error message showing which groups have which counts
        counts_by_group = {}
        for group_key, count in folder_counts.items():
            if count not in counts_by_group:
                counts_by_group[count] = []
            game_type, mechanism_type = group_key
            counts_by_group[count].append(f"{mechanism_type}_{game_type}")

        error_msg = "Folder count mismatch across groups:\n"
        for count in sorted(counts_by_group.keys()):
            groups = counts_by_group[count]
            error_msg += f"  {count} folder(s): {', '.join(groups)}\n"
        error_msg += "All game+mechanism combinations must have the same number of folders."

        raise AssertionError(error_msg)

    return list(unique_counts)[0]


def validate_list_consistency(
    value_lists: list[list[str]],
    identifiers: list[str],
    group_key: tuple[str, str],
    list_name: str,
) -> list[str]:
    """Validate all entries have same list in same order."""
    if not value_lists or not identifiers:
        raise ValueError(f"Empty inputs for validation of {group_key}")

    reference_list = value_lists[0]
    reference_id = identifiers[0]

    for i, (value_list, identifier) in enumerate(
        zip(value_lists[1:], identifiers[1:]), start=1
    ):
        if value_list != reference_list:
            raise AssertionError(
                f"{list_name.capitalize()} mismatch in {group_key}:\n"
                f"  Entry 0 ({reference_id}):\n    {reference_list}\n"
                f"  Entry {i} ({identifier}):\n    {value_list}\n"
                f"  Note: Both content AND order must match."
            )

    return reference_list


def validate_dict_consistency(
    value_dicts: list[dict],
    identifiers: list[str],
    group_key: tuple[str, str],
    dict_name: str,
) -> dict:
    """Validate all entries have identical dictionary content."""
    if not value_dicts or not identifiers:
        raise ValueError(f"Empty inputs for validation of {group_key}")

    reference_dict = value_dicts[0]
    reference_id = identifiers[0]

    for i, (value_dict, identifier) in enumerate(
        zip(value_dicts[1:], identifiers[1:]), start=1
    ):
        if value_dict != reference_dict:
            raise AssertionError(
                f"{dict_name.capitalize()} mismatch in {group_key}:\n"
                f"  Entry 0 ({reference_id}):\n    {reference_dict}\n"
                f"  Entry {i} ({identifier}):\n    {value_dict}"
            )

    return reference_dict
