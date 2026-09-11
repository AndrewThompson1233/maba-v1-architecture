# Contributing to Maba v1

Contributions to Maba v1 are welcome. Follow these guidelines to ensure consistency, numerical correctness, and clean code standards.

## Development Guidelines

1. **Architecture Integrity**: Keep the parameter budget exact (101,183,744 parameters). Any changes to layer dimensions or heads must align across PyTorch and the C++ engine.
2. **Code Style**:
   - Python: Clean, concise code. Minimal commentary focused on non-obvious details.
   - C++: C++17 standard with AVX2 and OpenMP acceleration.
   - Formatting: No em-dashes or en-dashes in code or documentation.
3. **Tests**: All changes must pass the full validation suite:
   ```bash
   bash run_full_validation.sh
   ```

## Pull Request Workflow

1. Fork the repository and create a branch from `main`.
2. Make your changes and verify with local tests.
3. Submit a pull request with a concise description of your changes.
