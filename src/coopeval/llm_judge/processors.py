"""
Text processing utilities for batch operations and data handling.
"""

import asyncio
import time
from typing import Any, Callable

import pandas as pd
from tqdm.asyncio import tqdm


class TextProcessingError(RuntimeError):
    """Raised when a single text cannot be processed successfully."""

    def __init__(
        self,
        *,
        message: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.metadata = metadata or {}


class TextProcessor:
    """
    Handles individual text processing operations.
    """

    def __init__(self, judge):
        """
        Initialize text processor.

        Args:
            judge: LLMJudge instance
        """
        self.judge = judge

    async def process_text_async(
        self, text: str, metadata: dict | None = None
    ) -> dict[str, Any]:
        """
        Process a single text with optional metadata.

        Args:
            text: Text to classify
            metadata: Optional metadata to include in result

        Returns:
            Dictionary with classification results and metadata
        """
        if not text or len(text.strip()) == 0:
            result = {
                "Reasoning_behind_classification": "No text provided.",
                "Confidence": 0.0,
                "justification_type": "Other",
            }
        else:
            result = await self.judge.classify_text_async(text)

        if metadata:
            result.update(metadata)

        final_result = {
            "classification_explanation": result.get(
                "Reasoning_behind_classification", ""
            ),
            "classification_confidence": result.get("Confidence", 0.0),
            "classification_justification": result.get(
                "justification_type", "Other"
            ),
        }

        if metadata:
            final_result.update(metadata)

        return final_result

    def process_text(
        self, text: str, metadata: dict | None = None
    ) -> dict[str, Any]:
        """Synchronous wrapper for process_text_async."""
        return asyncio.run(self.process_text_async(text, metadata))


class BatchProcessor:
    """
    Handles batch processing of multiple texts with parallel execution.
    """

    def __init__(self, judge, max_workers: int = 4, batch_size: int = 10):
        """
        Initialize batch processor.

        Args:
            judge: LLMJudge instance
            max_workers: Number of parallel workers
            batch_size: Size of each processing batch
        """
        self.judge = judge
        self.max_workers = max_workers
        self.batch_size = batch_size
        self.text_processor = TextProcessor(judge)

    def process_texts(
        self,
        texts: list[str],
        metadata_list: list[dict] | None = None,
        progress_callback: Callable | None = None,
    ) -> list[dict[str, Any]]:
        """Synchronous wrapper for process_texts_async."""
        return asyncio.run(
            self.process_texts_async(texts, metadata_list, progress_callback)
        )

    async def process_texts_async(
        self,
        texts: list[str],
        metadata_list: list[dict] | None = None,
        progress_callback: Callable | None = None,
    ) -> list[dict[str, Any]]:
        """
        Process a list of texts in parallel.

        Args:
            texts: List of texts to process
            metadata_list: Optional list of metadata dicts (same length as texts)
            progress_callback: Optional callback function for progress updates

        Returns:
            List of classification results
        """
        if not texts:
            return []

        if metadata_list is None:
            metadata_list = [{"index": i} for i in range(len(texts))]
        elif len(metadata_list) != len(texts):
            raise ValueError("metadata_list must have same length as texts")

        print(
            f"Processing {len(texts)} texts with {self.max_workers} workers"
        )

        start_time = time.time()
        semaphore = asyncio.Semaphore(self.max_workers)

        async def process_item(text: str, metadata: dict) -> dict[str, Any]:
            async with semaphore:
                try:
                    result = await self.text_processor.process_text_async(text, metadata)
                    if progress_callback:
                        progress_callback(1, len(texts))
                    return result
                except Exception as exc:
                    error_index = metadata.get("index")
                    raise TextProcessingError(
                        message=f"Failed to process text at index {error_index}",
                        metadata=metadata,
                    ) from exc

        tasks = [
            process_item(text, metadata)
            for text, metadata in zip(texts, metadata_list)
        ]

        all_results = []
        for f in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Processing"):
            all_results.append(await f)

        all_results.sort(key=lambda x: x.get("index", 0))

        elapsed = time.time() - start_time
        print(
            f"Completed processing {len(texts)} texts in {elapsed:.2f} seconds"
        )
        print(f"Average speed: {len(texts) / elapsed:.2f} texts/second")

        return all_results

    def process_dataframe(
        self,
        df: pd.DataFrame,
        text_column: str,
        metadata_columns: list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Process texts from a pandas DataFrame.

        Args:
            df: Input DataFrame
            text_column: Name of column containing texts to classify
            metadata_columns: List of columns to include as metadata

        Returns:
            DataFrame with classification results added
        """
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in DataFrame")

        texts = df[text_column].fillna("").astype(str).tolist()

        metadata_list = []
        for i, row in df.iterrows():
            metadata = {"index": i, "original_index": row.name}

            if metadata_columns:
                for col in metadata_columns:
                    if col in df.columns:
                        metadata[col] = row[col]

            metadata_list.append(metadata)

        results = self.process_texts(texts, metadata_list)

        results_df = pd.DataFrame(results)

        results_df = results_df.sort_values("original_index")

        for col in [
            "classification_explanation",
            "classification_confidence",
            "classification_justification",
        ]:
            if col in results_df.columns:
                df[col] = results_df[col].values

        return df
