"""
Taxonomy management for LLM Judge classifications.
"""

import json


class Taxonomy:
    """
    Manages classification taxonomies with categories and definitions.
    """

    def __init__(self, categories: dict[str, str] | None = None):
        """
        Initialize taxonomy.

        Args:
            categories: Dictionary mapping category names to definitions
        """
        self.categories = categories or {}

    @classmethod
    def from_json_file(cls, filepath: str) -> "Taxonomy":
        """
        Load taxonomy from JSON file.

        Args:
            filepath: Path to JSON file containing taxonomy

        Returns:
            Taxonomy instance
        """
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        return cls(categories=data.get("categories", {}))

    @classmethod
    def from_dict(cls, data: dict) -> "Taxonomy":
        """
        Create taxonomy from dictionary.

        Args:
            data: Dictionary containing taxonomy data

        Returns:
            Taxonomy instance
        """
        return cls(categories=data.get("categories", {}))

    def add_category(self, name: str, definition: str):
        """
        Add a category to the taxonomy.

        Args:
            name: Category name
            definition: Category definition
        """
        self.categories[name] = definition

    def remove_category(self, name: str):
        """
        Remove a category from the taxonomy.

        Args:
            name: Category name to remove
        """
        if name in self.categories:
            del self.categories[name]

    def get_categories(self) -> list[str]:
        """
        Get list of all category names.

        Returns:
            List of category names
        """
        return list(self.categories.keys())

    def get_definition(self, category: str) -> str | None:
        """
        Get definition for a specific category.

        Args:
            category: Category name

        Returns:
            Category definition or None if not found
        """
        return self.categories.get(category)

    def get_formatted_taxonomy(self) -> str:
        """
        Get formatted taxonomy text for use in prompts.

        Returns:
            Formatted taxonomy string
        """
        if not self.categories:
            return "Taxonomy:\n(No categories defined)"

        taxonomy_text = "Taxonomy:\n"
        for i, (category, definition) in enumerate(self.categories.items(), 1):
            taxonomy_text += f"{i}. {category}: {definition}\n"

        return taxonomy_text

    def to_dict(self) -> dict:
        """
        Convert taxonomy to dictionary format.

        Returns:
            Dictionary representation of taxonomy
        """
        return {"categories": self.categories}

    def save_to_json(self, filepath: str):
        """
        Save taxonomy to JSON file.

        Args:
            filepath: Path to save JSON file
        """
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    def __len__(self) -> int:
        """Return number of categories."""
        return len(self.categories)

    def __contains__(self, category: str) -> bool:
        """Check if category exists in taxonomy."""
        return category in self.categories
