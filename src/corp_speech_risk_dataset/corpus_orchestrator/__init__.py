"""
Corpus Orchestrator Module

This module provides thin orchestration for the complete corpus pipeline.
It coordinates the execution of all corpus modules via their CLIs.

Main components:
- End-to-end pipeline orchestration
- Demo and paper reproduction workflows
- Smoke testing across all modules
- Configuration management
"""

from .orchestrator import CorpusOrchestrator

__all__ = ["CorpusOrchestrator"]
