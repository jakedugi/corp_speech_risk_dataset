"""
Base extractor abstract class for data processing.
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional


class BaseExtractor(ABC):
    """
    Abstract base class for all data extractors.
    Extractors process fetched data and return structured results.
    """

    @abstractmethod
    def extract(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract structured data from raw input.

        Args:
            data: Raw data item to process

        Returns:
            List of extracted structured data items
        """
        pass

    @abstractmethod
    def validate(self, extracted_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Validate and filter extracted data.

        Args:
            extracted_data: List of extracted data items

        Returns:
            List of validated data items
        """
        pass

    def process_batch(self, data_batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process a batch of data items.

        Args:
            data_batch: List of raw data items

        Returns:
            List of all extracted and validated items
        """
        results = []
        for item in data_batch:
            extracted = self.extract(item)
            validated = self.validate(extracted)
            results.extend(validated)
        return results
