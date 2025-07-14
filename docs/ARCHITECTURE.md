# Corporate Speech Risk Dataset - Hexagonal + DDD Architecture

## Overview

This project has been reorganized to follow **Hexagonal Architecture** (Ports and Adapters) combined with **Domain-Driven Design (DDD)** principles. The structure follows Clean Architecture concentric circles where **outer layers depend only on inner layers**.

## Layer Structure

### 1. Domain Layer (`domain/`)
**Pure business objects, value types, NO dependencies**
- `quote_candidate.py` - Core business entity for quote processing
- `base_types.py` - Value objects and domain types
- `risk_features.py` - Risk calculation business logic (extracted from encoding)

**Principles:**
- No external dependencies
- Pure business logic only
- Innermost layer of Clean Architecture

### 2. Application Layer (`application/`)
**Use-cases & services orchestrating domain logic**
- `quote_extraction_pipeline.py` - Core quote extraction use-case
- `quote_extraction_config.py` - Configuration for extraction
- `courtlistener_orchestrator.py` - Legal data orchestration
- Various `run_*.py` - Use-case runners and workflows

**Principles:**
- Depends only on domain layer
- Defines interfaces for adapters (ports)
- Orchestrates domain objects
- No framework details

### 3. Adapters Layer (`adapters/`)
**Translate between outside world & application**

#### Inbound Adapters (`adapters/inbound/`)
- `cli.py` - Main CLI interface (Typer commands)
- `cli_encode.py` - Encoding CLI interface

#### Outbound Adapters (`adapters/outbound/`)
- `encoding/` - Text encoding and processing adapters
- `extractors/` - Data extraction adapters

**Principles:**
- Implement application layer interfaces
- Translate external formats to/from domain objects
- No business logic (only translation)

### 4. Infrastructure Layer (`infrastructure/`)
**HTTP, file-I/O, DB, spaCy, GPT, etc.**
- `api/` - HTTP client implementations
- `http_utils.py` - HTTP utility functions
- `file_io.py` - File system operations
- `nlp.py` - NLP framework integrations
- `nltk_setup.py` - NLTK configuration
- `resources/` - External data dependencies

**Principles:**
- Framework-specific code
- External system implementations
- Outermost layer of Clean Architecture

### 5. Shared Layer (`shared/`)
**Cross-cutting concerns (logging, settings)**
- `logging_utils.py` - Logging configuration and utilities
- `config.py` - Application configuration
- `constants.py` - Application constants
- `discovery.py` - Discovery utilities
- `stage_writer.py` - Stage writing utilities

**Principles:**
- No business logic
- Pure utilities and helpers
- Can be used by any layer

## Migration Summary

The following files were moved from their original locations:

### Domain Layer
- `models/quote_candidate.py` → `domain/quote_candidate.py`
- `custom_types/base_types.py` → `domain/base_types.py`
- Risk math from `encoding/wl_features.py` → `domain/risk_features.py`

### Application Layer
- `orchestrators/quote_extraction_pipeline.py` → `application/quote_extraction_pipeline.py`
- All orchestrator and workflow files → `application/`

### Adapters Layer
- `cli.py, cli_encode.py` → `adapters/inbound/`
- `encoding/` → `adapters/outbound/encoding/`
- `extractors/` → `adapters/outbound/extractors/`

### Infrastructure Layer
- `api/` → `infrastructure/api/`
- `utils/http_utils.py, utils/file_io.py` → `infrastructure/`
- `utils/nlp.py, utils/nltk_setup.py` → `infrastructure/`
- `resources/` → `infrastructure/resources/`

### Shared Layer
- `utils/logging_utils.py` → `shared/logging_utils.py`
- `config.py` → `shared/config.py`
- `utils/constants.py, utils/discovery.py, utils/stage_writer.py` → `shared/`

## Clean Architecture Rules

1. **Dependencies point inward only**:
   - `domain/` ← `application/` ← `adapters/` ← `infrastructure/`
   - `shared/` can be used by any layer

2. **Layer responsibilities**:
   - **Domain**: Pure business logic, no dependencies
   - **Application**: Use-cases, depends only on domain
   - **Adapters**: Interface translation, depends on domain + application
   - **Infrastructure**: External concerns, depends on all inner layers
   - **Shared**: Cross-cutting utilities, no business logic

3. **Original structure preserved**: All original folders remain unchanged for backward compatibility

## Benefits

- **Testability**: Domain logic can be tested without external dependencies
- **Flexibility**: Easy to swap out infrastructure components
- **Maintainability**: Clear separation of concerns
- **Independence**: Business logic is independent of frameworks and databases

Author: Jake Dugan <jake.dugan@ed.ac.uk>
Date: July 2024
