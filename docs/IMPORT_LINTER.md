# Import Linter - Hexagonal + DDD Architecture Enforcement

## Overview

This project uses [Import Linter](https://github.com/seddonym/import-linter) to automatically enforce the hexagonal architecture + Domain-Driven Design (DDD) layered structure. The linter ensures that code follows Clean Architecture dependency rules where **outer layers can only import from inner layers**.

## Layer Hierarchy

The layers are organized from most permissive (outer) to most restrictive (inner):

1. **`shared`** - Cross-cutting utilities (can be used by any layer)
2. **`infrastructure`** - External systems, frameworks, I/O
3. **`adapters`** - Interface translation (inbound/outbound)
4. **`application`** - Use-cases and orchestration
5. **`domain`** - Pure business logic (no external dependencies)

## Dependency Rules

**Allowed imports:**
- `infrastructure` → `adapters`, `application`, `domain`, `shared`
- `adapters` → `application`, `domain`, `shared`
- `application` → `domain`, `shared`
- `domain` → `shared` only
- `shared` → no internal dependencies

**Forbidden imports:**
- Any inner layer importing from outer layers
- `domain` importing from `application`, `adapters`, or `infrastructure`
- `application` importing from `adapters` or `infrastructure`

## Running Locally

```bash
# Install Import Linter
pip install import-linter

# Run from project root
cd src
PYTHONPATH=. lint-imports --config ../.importlinter
```

## CI Integration

The linter runs automatically in CI as part of the `architecture-check` job. It will fail the build if any layer violates the dependency rules.

## Configuration

The configuration is in `.importlinter` at the project root:

```ini
[importlinter]
root_package = corp_speech_risk_dataset
exclude_type_checking_imports = True

[importlinter:contract:layers]
name = Corporate Speech Risk Dataset Layered Architecture
type = layers
layers =
    corp_speech_risk_dataset.shared
    corp_speech_risk_dataset.infrastructure
    corp_speech_risk_dataset.adapters
    corp_speech_risk_dataset.application
    corp_speech_risk_dataset.domain
exhaustive = False
```

## Troubleshooting

**"Missing layer" error:** Make sure all layer directories have `__init__.py` files.

**"Could not find package" error:** Ensure you're running from the `src/` directory with `PYTHONPATH=.`

**Import violations:** Check that your imports follow the dependency rules above. Move business logic down to appropriate layers.

## Benefits

- **Prevents architecture erosion** - Catches violations early
- **Enforces separation of concerns** - Keeps business logic isolated
- **Improves testability** - Domain logic can be tested without dependencies
- **Documents architecture** - Makes layer boundaries explicit
