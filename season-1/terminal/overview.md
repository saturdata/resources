# Terminal and Git Basics Overview

## Discussion Topics

- **Why terminal matters for data scientists**
  - Essential for testing code like an engineer
  - Required for CLI (command-line interface) tools used in data work
  - Dramatically faster than GUI operations for many tasks
  - Industry standard for development workflows
  - Gateway to powerful automation and scripting capabilities

- **Core navigation commands**
  - `pwd` - Print working directory (know where you are)
  - `ls` - List directory contents (see what's available)
  - `ls -la` - List with details and hidden files (complete picture)
  - `cd <directory>` - Change to directory (move around)
  - `cd ..` - Go up one directory level (navigate backwards)
  - `cd ~` - Go to home directory (quick return home)

- **Essential file operations**
  - `mkdir <name>` - Create directory (make new folders)
  - `touch <file>` - Create empty file (quick file creation)
  - `cp <source> <dest>` - Copy file/directory (duplicate content)
  - `mv <source> <dest>` - Move/rename file (relocate or rename)
  - `rm <file>` - Delete file (remove single files)
  - `rm -rf <directory>` - Delete directory recursively (remove folders)
  - `open <file>` - Open file with default app (launch files)

- **Text editing with vim basics**
  - `vim <filename>` - Open file in vim editor
  - `i` - Enter insert mode (start editing)
  - `Esc` - Return to normal mode (stop editing)
  - `:wq` - Write and quit (save and exit)
  - `:q!` - Quit without saving (abandon changes)
  - Alternative: `nano <filename>` for beginner-friendly editing

- **Terminal productivity tips**
  - Tab completion for filenames and directories
  - Up/down arrows for command history navigation
  - `Ctrl+C` to cancel running commands
  - `Cmd+K` or `clear` to clean up the screen
  - Combine commands with `&&` for sequential execution

- **The problem Git solves**
  - Traditional storage (Teams folders, Excel) lacks change tracking
  - No way to recover previous versions or see what changed
  - Collaboration becomes messy without proper version control
  - Git provides complete history and branching capabilities
  - Enables professional development workflows

- **Core Git concepts**
  - **Repository (repo)** - A project folder tracked by Git
  - **Clone** - Download a copy of a remote repository
  - **Stage** - Prepare changes to be committed
  - **Commit** - Save changes with a descriptive message
  - **Push** - Upload local changes to remote repository
  - **Pull** - Download changes from remote repository

- **Essential Git workflow**
  - `gh repo clone <repository>` - Clone repository using GitHub CLI
  - `git status` - Check current repository state
  - `git add .` - Stage all changes (or `git add <filename>` for specific files)
  - `git commit -m "message"` - Commit changes with descriptive message
  - `git push` - Push changes to remote repository
  - `git pull` - Pull latest changes from remote

- **Git best practices**
  - Write clear, descriptive commit messages
  - Commit frequently with logical, focused changes
  - Always pull before starting new work
  - Use `git status` regularly to understand current state
  - Stage and review changes before committing

- **Python environment management challenges**
  - Different projects require different Python packages
  - Package versions can conflict between projects
  - System-wide installations break when updated
  - Sharing reproducible environments with team members
  - Managing dependencies across development and production

- **Virtual environments (venvs) solution**
  - Isolated Python environments per project
  - No dependency conflicts between projects
  - Reproducible, shareable environment specifications
  - Traditional activation: `source venv_name/bin/activate`

- **Modern solution: uv**
  - Lightning-fast Python package and project manager
  - Automatic dependency resolution and virtual environment creation
  - Dynamic virtual environment creation on command execution
  - Seamless integration with marimo notebooks
  - Eliminates manual venv management overhead

- **Running Python code professionally**
  - `python script.py` - Basic execution (not recommended for projects)
  - `python3 script.py` - Explicit Python 3 usage
  - `uv run python script.py` - Recommended for consistent environments
  - `uv run marimo run notebook.py` - Execute marimo notebooks with dependencies

- **Marimo notebook advantages**
  - Modern Python notebook framework
  - Reactive execution (cells update automatically when dependencies change)
  - Better engineering features than Jupyter
  - Git-friendly (plain Python files, not JSON)
  - Excellent integration with uv for dependency management
  - Professional development workflow compatibility

- **Common troubleshooting scenarios**
  - Permission denied errors and file permissions (`ls -la`, `chmod`)
  - Command not found issues (PATH problems, tool installation)
  - Git authentication failures (GitHub CLI setup)
  - Python/uv package installation problems (updates, connectivity)
  - Terminal customization for improved productivity

- **Next-level terminal setup (optional)**
  - iTerm2 for advanced terminal features
  - Starship prompt for Git integration and status indicators
  - Warp terminal for modern, AI-enhanced experience
  - Color schemes like Catppuccin for visual appeal
  - Custom aliases and shell functions for efficiency