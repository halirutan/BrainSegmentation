# Contributing to Brain Segmentation

Thank you for your interest in contributing to the Brain Segmentation project! This document provides guidelines and instructions for contributing.

## Code of Conduct

Please be respectful and considerate of others when contributing to this project. We aim to foster an inclusive and welcoming community.

## How to Contribute

There are many ways to contribute to this project:

1. **Report bugs**: If you find a bug, please create an issue in our GitHub repository with a clear description of the problem, steps to reproduce it, and your environment details.

2. **Suggest features**: If you have ideas for new features or improvements, please create an issue describing your suggestion.

3. **Submit code changes**: If you'd like to fix a bug or implement a feature, you can submit a pull request.

4. **Improve documentation**: Help us improve our documentation by fixing errors, adding examples, or clarifying explanations.

## Development Setup

To set up your development environment:

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/yourusername/BrainSegmentation.git
   cd BrainSegmentation
   ```
3. Create a virtual environment and install dependencies:
   ```bash
   conda env create -f python_setup/environment.yml
   conda activate brainseg
   ```
4. Create a branch for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Pull Request Process

1. Ensure your code follows our coding standards
2. Add or update tests as necessary
3. Update documentation to reflect any changes
4. Make sure all tests pass
5. Submit a pull request with a clear description of the changes

## Coding Standards

- Follow PEP 8 style guidelines for Python code
- Write docstrings for all functions, classes, and modules
- Include type hints where appropriate
- Write unit tests for new functionality

## Testing

Before submitting a pull request, make sure all tests pass:

```bash
# Run tests
pytest
```

## Documentation

When adding new features or making changes, please update the relevant documentation:

- Update docstrings in the code
- Update or add examples if necessary
- Update the relevant markdown files in the `docs` directory

## License

By contributing to this project, you agree that your contributions will be licensed under the same license as the project (see [LICENSE](license.md)).

## Questions?

If you have any questions about contributing, please open an issue or contact the project maintainers.

Thank you for your contributions!