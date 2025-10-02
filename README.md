# Saturdata resources

Welcome to the resources repo for [Saturdata](https://saturdata.github.io), your favorite weekend data podcast. Check out our resources on analytics, data science, and more!

## Setup

### Prerequisites

- Python 3.11 or higher
- [Git](https://git-scm.com/)
  - Install [here](https://git-scm.com/downloads)
- [GitHub CLI](https://cli.github.com/)
  - Install with `brew install gh`
- [uv](https://docs.astral.sh/uv/)
  - Install with `curl -LsSf https://astral.sh/uv/install.sh | sh`

### Setup

1. Clone the repository:
   ```bash
   gh repo clone saturdata/resources
   cd resources
   ```

1. Install dependencies and run the [Marimo](https://marimo.io/) notebook:
   ```bash
   uv run marimo run season-1/analytics_basics.py
   ```

That's it! The `uv run` command will automatically:
- Create a virtual environment
- Install all required dependencies
- Launch the marimo notebook

To edit your local copy of the notebook, run:
   ```bash
   uv run marimo edit season-1/analytics_basics.py
   ```

## Contribute

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request
