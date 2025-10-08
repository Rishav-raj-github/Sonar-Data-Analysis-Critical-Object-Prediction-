# Contributing to Sonar Data Analysis

Thank you for your interest in contributing to the Sonar Data Analysis for Critical Object Prediction project! We welcome contributions from the community and appreciate your efforts to improve this project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Coding Standards](#coding-standards)
- [Pull Request Process](#pull-request-process)
- [Branching Strategy](#branching-strategy)
- [Testing Guidelines](#testing-guidelines)
- [Documentation](#documentation)
- [Troubleshooting](#troubleshooting)
- [Community Guidelines](#community-guidelines)

## Code of Conduct

We are committed to providing a welcoming and inclusive environment for all contributors. Please:

- Be respectful and considerate in all interactions
- Welcome newcomers and help them get started
- Focus on constructive feedback
- Respect differing viewpoints and experiences
- Show empathy towards other community members

## Getting Started

### Prerequisites

- Python 3.7 or higher
- Git for version control
- A GitHub account
- Familiarity with machine learning concepts (recommended)

### Setting Up Your Development Environment

1. **Fork the repository** to your GitHub account

2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/Sonar-Data-Analysis-Critical-Object-Prediction-.git
   cd Sonar-Data-Analysis-Critical-Object-Prediction-
   ```

3. **Add the upstream remote**:
   ```bash
   git remote add upstream https://github.com/Rishav-raj-github/Sonar-Data-Analysis-Critical-Object-Prediction-.git
   ```

4. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

5. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install pytest pytest-cov flake8 black pylint  # Development dependencies
   ```

## Development Workflow

1. **Sync with upstream** before starting new work:
   ```bash
   git fetch upstream
   git checkout main
   git merge upstream/main
   ```

2. **Create a new branch** for your changes (see [Branching Strategy](#branching-strategy))

3. **Make your changes** following our [Coding Standards](#coding-standards)

4. **Test your changes** thoroughly

5. **Commit your changes** with clear, descriptive messages

6. **Push to your fork** and create a pull request

## Coding Standards

### Style Guide

We strictly follow **PEP 8** for Python code style. All contributions must adhere to these standards.

#### Code Formatting

**Required**: Use `flake8` for linting and `black` for automatic formatting.

```bash
# Format your code with black (line length: 88)
black sonar_analysis.py

# Check for style violations with flake8
flake8 sonar_analysis.py --max-line-length=88 --extend-ignore=E203,W503
```

**Black Configuration**:
- Line length: 88 characters (black default)
- String quotes: Double quotes preferred
- Trailing commas: Enabled for multi-line structures

**Flake8 Configuration**:
- Max line length: 88
- Ignore: E203 (whitespace before ':'), W503 (line break before binary operator)

#### Documentation Standards

All functions, classes, and modules must have **Google-style docstrings**.

**Example**:
```python
def predict_object(model, input_data: tuple) -> str:
    """Predict whether an object is a rock or mine based on sonar data.
    
    Args:
        model: Trained logistic regression model.
        input_data (tuple): A tuple of 60 numerical features representing
            sonar frequency band energies.
    
    Returns:
        str: Prediction result - either 'Rock' or 'Mine'.
    
    Raises:
        ValueError: If input_data doesn't contain exactly 60 features.
    
    Example:
        >>> model = train_model(X_train, y_train)
        >>> result = predict_object(model, (0.0124, 0.0433, ..., 0.0062))
        >>> print(result)
        'Rock'
    """
    # Implementation
    pass
```

#### Type Hints

**Required**: Add type hints to all function signatures.

```python
from typing import Tuple, Any
import pandas as pd
from sklearn.linear_model import LogisticRegression

def prepare_data(
    data: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Prepare and split data for training and testing."""
    pass
```

### Code Quality Checklist

Before submitting your PR, ensure:

- [ ] Code is formatted with `black`
- [ ] Code passes `flake8` linting with no errors
- [ ] All functions have Google-style docstrings
- [ ] Type hints are added to function signatures
- [ ] Code follows PEP 8 conventions
- [ ] No unused imports or variables
- [ ] Appropriate error handling is implemented

## Pull Request Process

### Before Submitting

1. **Run all tests** and ensure they pass:
   ```bash
   pytest tests/ -v
   ```

2. **Run linting checks**:
   ```bash
   flake8 sonar_analysis.py
   black --check sonar_analysis.py
   ```

3. **Update documentation** if you've made changes to:
   - Function signatures
   - Module behavior
   - Dependencies
   - Configuration

4. **Add tests** for new functionality

### Creating Your Pull Request

1. **Use a clear, descriptive title**:
   - Good: "Add cross-validation support for model evaluation"
   - Bad: "Update code"

2. **Provide a detailed description** including:
   - What changes were made
   - Why the changes were necessary
   - How to test the changes
   - Screenshots (if applicable)
   - Related issues (use "Fixes #123" or "Closes #123")

3. **Use the PR template** (if provided)

4. **Request review** from maintainers

### Review Expectations

- **Response time**: Expect initial feedback within 2-3 business days
- **Iteration**: Be prepared to make changes based on feedback
- **Discussion**: Feel free to discuss and explain your approach
- **Patience**: Complex PRs may take longer to review

### After Review

- Address all review comments
- Request re-review when ready
- Keep the PR updated with the main branch
- Be responsive to additional feedback

## Branching Strategy

We use a simple, organized branching model:

### Branch Naming Convention

**Format**: `<type>/<short-description>`

**Types**:
- `feature/` - New features or enhancements
- `bugfix/` - Bug fixes
- `hotfix/` - Critical fixes for production issues
- `docs/` - Documentation-only changes
- `refactor/` - Code refactoring without behavior changes
- `test/` - Adding or updating tests

**Examples**:
```bash
git checkout -b feature/add-svm-classifier
git checkout -b bugfix/fix-data-loading-error
git checkout -b docs/update-installation-guide
git checkout -b refactor/improve-model-training
```

### Branch Guidelines

- **Keep branches focused**: One feature/fix per branch
- **Keep branches up-to-date**: Regularly merge main into your branch
- **Delete after merge**: Clean up branches after PR is merged
- **Short-lived**: Complete work within 1-2 weeks when possible

## Testing Guidelines

### Running Tests

```bash
# Run all tests
pytest tests/

# Run with verbose output
pytest tests/ -v

# Run with coverage report
pytest --cov=sonar_analysis tests/

# Run specific test file
pytest tests/test_sonar_analysis.py
```

### Writing Tests

- **Test all new functionality**: Every new feature should have tests
- **Use descriptive names**: `test_predict_object_returns_rock_for_rock_features`
- **Follow AAA pattern**: Arrange, Act, Assert
- **Test edge cases**: Empty inputs, invalid data, boundary conditions
- **Maintain coverage**: Aim for >80% code coverage

**Example Test**:
```python
import pytest
from sonar_analysis import load_sonar_data

def test_load_sonar_data_returns_dataframe():
    """Test that load_sonar_data returns a pandas DataFrame."""
    # Arrange
    filename = 'Copy of sonar data.csv'
    
    # Act
    result = load_sonar_data(filename, header=None)
    
    # Assert
    assert isinstance(result, pd.DataFrame)
    assert result.shape == (208, 61)  # Expected dimensions
```

### Coverage Requirements

- Maintain minimum **80% code coverage**
- Critical functions should have **>90% coverage**
- All new code must include tests

## Documentation

### What to Document

- All public functions and classes
- Module-level docstrings
- Complex algorithms or logic
- Configuration options
- Installation and setup instructions
- Examples and tutorials

### Documentation Style

- Use **Google-style docstrings** for all Python code
- Use **Markdown** for all `.md` files
- Keep examples up-to-date with code changes
- Include type information in docstrings

### Updating README

If your changes affect user-facing functionality, update:
- Installation instructions
- Usage examples
- API documentation
- Feature list

## Troubleshooting

### Common Issues

#### Import Errors

**Problem**: `ModuleNotFoundError` when running code

**Solution**:
```bash
# Ensure virtual environment is activated
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Reinstall dependencies
pip install -r requirements.txt
```

#### Test Failures

**Problem**: Tests fail locally but pass in CI

**Solution**:
- Ensure you're using the same Python version as CI (3.7+)
- Check for environment-specific dependencies
- Clear pytest cache: `pytest --cache-clear`

#### Flake8 Errors

**Problem**: Flake8 reports style violations

**Solution**:
```bash
# Automatically fix most issues with black
black sonar_analysis.py

# Check remaining issues
flake8 sonar_analysis.py --max-line-length=88
```

#### Merge Conflicts

**Problem**: Git reports merge conflicts when updating branch

**Solution**:
```bash
# Update your main branch
git checkout main
git pull upstream main

# Rebase your feature branch
git checkout feature/your-feature
git rebase main

# Resolve conflicts in your editor, then:
git add <resolved-files>
git rebase --continue
```

### Getting Help

If you're stuck:

1. **Check existing issues**: Someone may have encountered the same problem
2. **Review documentation**: README, docstrings, and comments
3. **Ask in discussions**: Use GitHub Discussions for questions
4. **Open an issue**: If you've found a bug or have a question

## Community Guidelines

### Communication Etiquette

- **Be respectful**: Treat all community members with kindness and respect
- **Be patient**: Remember that maintainers and reviewers are often volunteers
- **Be clear**: Provide sufficient context in questions and issues
- **Be constructive**: Focus on solutions rather than problems
- **Be collaborative**: Work together to find the best solutions

### Issue Reporting

When reporting bugs:
- Use a clear, descriptive title
- Provide steps to reproduce
- Include error messages and stack traces
- Specify your environment (OS, Python version, etc.)
- Include relevant code snippets

### Feature Requests

When requesting features:
- Explain the use case and benefits
- Provide examples of how it would work
- Consider alternative approaches
- Be open to discussion and iteration

### Code Reviews

When reviewing others' code:
- Be constructive and specific
- Explain the reasoning behind suggestions
- Acknowledge good work
- Focus on the code, not the person
- Use "we" instead of "you" (e.g., "We could improve this by...")

## Recognition

All contributors will be:
- Listed in the project's contributors list
- Credited in release notes for significant contributions
- Welcomed as valued community members

## Questions?

If you have questions about contributing:
- Open a [GitHub Discussion](https://github.com/Rishav-raj-github/Sonar-Data-Analysis-Critical-Object-Prediction-/discussions)
- Review existing issues and pull requests
- Contact the maintainers

---

**Thank you for contributing to Sonar Data Analysis! Your efforts help make this project better for everyone.** 🎉
