# Terminal and Git Basics for Data Scientists

This task file provides a comprehensive guide for creating terminal and git resources for data scientists, analysts, and engineers.

## Task Overview

Create a folder in `season-1` dir called `terminal` containing:
- `overview.md` with bullet points covering terminal and git basics
- `resource.py` marimo notebook with practical examples

## Prerequisites

Before starting, ensure the following tools are installed on macOS:

### Required Tools
1. **Homebrew** (package manager for macOS)
   - Install: `/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"`
   - Verify: `brew --version`

2. **Git** (version control system)
   - Install: `brew install git`
   - Verify: `git --version`
   - Configure: 
     ```bash
     git config --global user.name "Your Name"
     git config --global user.email "your.email@example.com"
     ```

3. **GitHub CLI** (for easier GitHub operations)
   - Install: `brew install gh`
   - Verify: `gh --version`
   - Authenticate: `gh auth login`

4. **uv** (Python package and project manager)
   - Install: `brew install uv`
   - Verify: `uv --version`

## Content Structure

### Terminal Basics Section

#### Why Learn Terminal?
- Essential for testing code like an engineer
- Required for CLI (command-line interface) tools used in data work
- Dramatically faster than GUI operations for many tasks
- Industry standard for development workflows

#### Getting Started
- **Opening Terminal**: Press `Cmd+Space`, search "Terminal", press Enter
- **Alternative terminals**: iTerm2, Warp (mentioned in customization section)

Mention that now is the time to install all the prerequisites listed above in your terminal

#### Core Navigation Commands
| Command | Description | GUI Equivalent |
|---------|-------------|----------------|
| `pwd` | Print working directory | See current folder in Finder |
| `ls` | List directory contents | View files in Finder |
| `ls -la` | List with details and hidden files | Show hidden files in Finder |
| `cd <directory>` | Change to directory | Double-click folder in Finder |
| `cd ..` | Go up one directory level | Click back button in Finder |
| `cd ~` | Go to home directory | Navigate to user folder |

#### File Operations
| Command | Description | GUI Equivalent |
|---------|-------------|----------------|
| `mkdir <name>` | Create directory | Right-click → New Folder |
| `touch <file>` | Create empty file | Right-click → New Document |
| `cp <source> <dest>` | Copy file/directory | Cmd+C, Cmd+V |
| `mv <source> <dest>` | Move/rename file | Drag and drop |
| `rm <file>` | Delete file | Move to Trash |
| `rm -rf <directory>` | Delete directory recursively | Move folder to Trash |
| `open <file>` | Open file with default app | Double-click file |

#### Text Editing with vim
- `vi <filename>` or `vim <filename>`: Open file in vim editor
- **Insert mode**: Press `i` to start editing
- **Normal mode**: Press `Esc` to exit insert mode
- **Save and quit**: `:wq` (write and quit)
- **Quit without saving**: `:q!`
- **Alternative**: Use `nano <filename>` for a more user-friendly editor

#### Tips and Tricks
- **Tab completion**: Start typing a filename/directory and press Tab
- **Command history**: Use up/down arrow keys to navigate previous commands
- **Cancel command**: Press `Ctrl+C`
- **Clear screen**: Press `Cmd+K` or type `clear`

### Version Control with Git

#### The Problem Git Solves
- Traditional storage methods don't track changes
- No way to recover previous versions
- Collaboration becomes messy without proper version control
- Git provides a complete history of all changes

#### Core Git Concepts
- **Repository (repo)**: A project folder tracked by Git
- **Clone**: Download a copy of a remote repository
- **Stage**: Prepare changes to be committed
- **Commit**: Save changes with a descriptive message
- **Push**: Upload local changes to remote repository
- **Pull**: Download changes from remote repository

#### Essential Git Workflow

1. **Clone a repository**:
   ```bash
   gh repo clone saturdata/resources
   cd resources
   ```

2. **Check repository status**:
   ```bash
   git status
   ```

3. **Stage changes**:
   ```bash
   git add .                    # Stage all changes
   git add <filename>           # Stage specific file
   ```

4. **Commit changes**:
   ```bash
   git commit -m "Descriptive commit message"
   ```

5. **Push to remote**:
   ```bash
   git push
   ```

6. **Pull latest changes**:
   ```bash
   git pull
   ```

#### Best Practices
- Write clear, descriptive commit messages. Learn about conventional commits here: https://www.conventionalcommits.org/en/v1.0.0/
- Commit frequently with logical changes
- Pull before starting new work
- Use `git status` often to check your current state

### Python Environment Management

#### The Dependency Problem
- Different projects require different Python packages
- Package versions can conflict between projects
- System-wide installations can break when updated
- Solution: Virtual environments isolate project dependencies

#### Virtual Environments (venvs)
- **Purpose**: Isolated Python environments per project
- **Benefits**: No dependency conflicts, reproducible environments
- **Traditional activation**: `source venv_name/bin/activate`

#### Modern Solution: uv
- **Advantages**: Lightning-fast, automatic dependency management
- **Key feature**: Dynamic virtual environment creation
- **Integration**: Works seamlessly with marimo notebooks

### Running Marimo Notebooks

#### What is Marimo?
- Modern Python notebook framework
- Alternative to Jupyter with better engineering features
- Reactive execution (cells update automatically)
- Works excellently with uv for dependency management

#### Step-by-Step Execution

1. **Navigate to repository**:
   ```bash
   cd resources
   ```

2. **Run a resource notebook**:
   ```bash
   uv run marimo run season-1/data-visualization/resource.py
   ```

3. **Alternative with relative paths**:
   ```bash
   cd season-1/data-visualization
   uv run marimo run resource.py
   ```

#### Command Breakdown
- `uv run`: Handles dependencies and creates virtual environment automatically
- `marimo run`: Starts the marimo web application locally
- `season-1/data-visualization/resource.py`: File path to the notebook

### Running Python Scripts

#### Basic Execution
```bash
python script.py        # Use system Python (not recommended)
python3 script.py       # Use Python 3 explicitly
uv run python script.py # Use uv-managed Python (recommended)
```

#### Best Practices
- Always use `uv run` for consistent environments
- Ensure script dependencies are properly declared
- Test scripts in isolated environments

## Terminal Customization (Optional)

For enhanced terminal experience:

### iTerm2
- Advanced terminal replacement for macOS Terminal
- Features: Split panes, better search, customization options
- Install: Download from https://iterm2.com/

### Starship Prompt
- Cross-shell prompt with Git integration and status indicators
- Install: `brew install starship`
- Configure: Add to shell profile (`.zshrc` for zsh)

### Warp Terminal
- Modern, collaborative terminal with AI features
- GPU-accelerated with built-in completions
- Download from https://www.warp.dev/

### Color Schemes
- **Catppuccin**: Popular pastel color scheme
- Apply to terminal, editors, and development tools
- Available at https://github.com/catppuccin

## Troubleshooting Common Issues

### Permission Denied
- **Problem**: Cannot execute commands or access files
- **Solution**: Check file permissions with `ls -la`, use `chmod` if needed

### Command Not Found
- **Problem**: Installed tool not recognized
- **Solution**: Check if tool is in PATH, restart terminal, or reinstall

### Git Authentication
- **Problem**: Cannot push to repository
- **Solution**: Ensure GitHub CLI is authenticated (`gh auth login`)

### Python/uv Issues
- **Problem**: Package installation failures
- **Solution**: Update uv (`brew upgrade uv`), check internet connection

## Conclusion

Terminal proficiency is essential for modern data work. You don't need to be an expert, but mastering these basics will:
- Enable you to use professional development tools
- Dramatically speed up common tasks
- Prepare you for collaborative data science workflows
- Allow you to leverage powerful command-line tools

Happy building! 🚀

## Learning Resources

- [Git Documentation](https://git-scm.com/doc)
- [uv Documentation](https://docs.astral.sh/uv/)
- [Marimo Documentation](https://docs.marimo.io/)
- [Terminal Cheat Sheet](https://github.com/0nn0/terminal-mac-cheatsheet)