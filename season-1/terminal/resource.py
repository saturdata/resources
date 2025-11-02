#!/usr/bin/env python3
"""
Terminal and Git Basics for Data Professionals - Marimo Notebook
=============================================================
Covers terminal navigation, file operations, git workflow, and Python environment management
"""

import marimo

__generated_with = "0.16.5"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md(
        r"""
    # Terminal and Git Basics for Data Professionals
    Saturdata: Season 1

    This notebook provides hands-on examples for mastering terminal and git fundamentals:
    - Terminal navigation and file operations
    - Git version control workflow
    - Python environment management with uv
    - Running marimo notebooks professionally
    - Troubleshooting common issues
    """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""## Why Terminal Matters""")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### The Power of Command Line

    As a data professional, mastering the terminal unlocks:
    - **Engineering workflows**: Test code like a professional developer
    - **Speed**: Navigate faster than clicking through folders
    - **Automation**: Chain commands together for repetitive tasks
    - **Tool access**: Many data science tools are CLI-first
    - **Reproducibility**: Document exact steps for others to follow

    Let's start with some examples you can try right now!
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Open terminal

    Use `Cmd+Space` to open spotlight search and type `terminal`. Press enter and you should see a black terminal window appear
    """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""## Essential Terminal Commands""")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Navigation Commands

    Try these commands in your terminal (open Terminal.app on Mac):

    ```bash
    # See where you are
    pwd

    # List files and folders
    ls

    # List with details (permissions, size, date)
    ls -la

    # Change to a directory
    cd Documents

    # Go back one level
    cd ..

    # Go to your home directory
    cd ~
    ```

    **Pro tip**: Use Tab completion! Start typing a filename and press Tab.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### File Operations

    Create, copy, move, and delete files like a pro:

    ```bash
    # Create a new directory
    mkdir my_data_project

    # Create an empty file
    touch analysis.py

    # Copy a file
    cp analysis.py backup_analysis.py

    # Move or rename a file
    mv analysis.py data_analysis.py

    # Delete a file (be careful!)
    rm backup_analysis.py

    # Delete a directory and everything in it (very careful!)
    rm -rf old_project_folder

    # Open a file with default application
    open data_analysis.py
    ```
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Terminal Productivity Tips

    Make your terminal work faster:

    ```bash
    # Use command history with arrow keys
    # Press ↑ to see previous commands

    # Cancel a running command
    # Press Ctrl+C

    # Clear the screen
    # Press Cmd+K or type:
    clear

    # Run multiple commands in sequence using `&&`
    mkdir new_project && cd new_project && touch README.md

    # See command history
    history
    ```
    """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""## Text Editing with vim""")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### vim Basics (Don't Panic!)

    vim is everywhere in data science environments. Here's the survival guide:

    ```bash
    # Open a file in vim
    vim my_script.py
    ```

    **Inside vim**:
    - Press `i` to enter INSERT mode (start typing)
    - Press `Esc` to return to NORMAL mode
    - Type `:wq` and Enter to save and quit
    - Type `:q!` and Enter to quit without saving

    **Alternative**: Use nano for a friendlier editor:
    ```bash
    nano my_script.py
    # Ctrl+X to exit, Y to save
    ```
    """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""## Version Control with Git""")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### The Problem Git Solves

    **Before Git**: 
    - `analysis_final.py`
    - `analysis_final_v2.py` 
    - `analysis_final_ACTUALLY_FINAL.py`
    - Files stored in Teams folders with no history

    **With Git**:
    - Complete history of every change
    - Collaborate without conflicts
    - Revert to any previous version
    - Professional development workflow
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Install Git and GitHub CLI
    ```bash
    brew install git
    brew install gh
    ```
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Essential Git Workflow

    This is the workflow you'll use every day:

    ```bash
    # 1. Clone a repository (download a project)
    gh repo clone saturdata/resources
    cd resources

    # 2. Check what's changed
    git status

    # 3. Stage your changes (prepare to save)
    git add .                    # Stage everything
    git add specific_file.py     # Stage just one file

    # 4. Commit with a message (save changes)
    git commit -m "Add data analysis for customer segments"

    # 5. Push to remote (upload changes)
    git push

    # 6. Get latest changes from team
    git pull
    ```
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Git Best Practices

    **Good commit messages**:
    - ✅ "Fix bug in data preprocessing pipeline"
    - ✅ "Add visualization for quarterly revenue trends"  
    - ❌ "updates"
    - ❌ "fix stuff"

    **Good workflow habits**:
    - Always `git pull` before starting work
    - Use `git status` frequently to see what's changed
    - Commit often with focused changes
    - Write commits like you're talking to your future self

    **Pro tip**: The best standard for Git commit messages is [conventional commits](https://www.conventionalcommits.org/en/v1.0.0/)!
    """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""## Python Environment Management""")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### The Dependency Problem

    **The Nightmare Scenario**:
    - Project A needs pandas 1.5.0
    - Project B needs pandas 2.0.0  
    - You install pandas 2.0.0 globally
    - Project A breaks! 💥

    **The Solution**: Virtual environments isolate dependencies per project.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Traditional Virtual Environments

    The old way (still works, but tedious):

    ```bash
    # Create a virtual environment
    python3 -m venv my_project_env

    # Activate it (you must do this every time!)
    source my_project_env/bin/activate

    # Install packages with pip (Python package manager)
    pip install pandas polars

    # Deactivate when done
    deactivate
    ```

    **Problem**: You have to remember to activate/deactivate!
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Modern Solution: uv

    uv handles everything automatically:

    ```bash
    # Install uv (one time setup)
    brew install uv

    # Run Python with automatic environment management
    uv run python my_script.py

    # Run marimo with dependencies handled automatically
    uv run marimo run my_notebook.py

    # Install packages for current project
    uv add pandas polars seaborn
    ```

    **Magic**: uv creates and manages environments behind the scenes!
    """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""## Running Python Code Professionally""")
    return


@app.cell
def _():
    import os
    import subprocess

    return (os,)


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Different Ways to Run Python

    From worst to best practices:
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ```bash
    # ❌ System Python (can break your system)
    python script.py

    # ⚠️ Better, but still not isolated
    python3 script.py

    # ✅ Best: uv handles everything
    uv run python script.py
    ```
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Running This Notebook

    Here's how you launched this marimo notebook:

    ```bash
    # Navigate to the resources directory
    cd resources

    # Run the notebook (uv handles all dependencies)
    uv run marimo run season-1/terminal/resource.py
    ```

    **What's happening**:
    - `uv run`: Creates virtual environment automatically
    - `marimo run`: Launches the notebook web application  
    - `season-1/terminal/resource.py`: Path to this file
    """
    )
    return


@app.cell
def _(mo, os):
    # Show current working directory and environment info
    current_dir = os.getcwd()
    mo.md(
        f"""
    ### Environment Information

    **Current directory**: `{current_dir}`

    **Python version**: Running in isolated environment managed by uv

    **Available packages**: marimo and dependencies automatically available
    """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""## Troubleshooting Common Issues""")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Permission Denied Errors

    **Problem**: `Permission denied` when running commands

    **Solutions**:
    ```bash
    # Check file permissions
    ls -la problematic_file.py

    # Make file executable
    chmod +x problematic_file.py

    # For directories, use recursive
    chmod -R 755 project_folder/
    ```
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Command Not Found

    **Problem**: `command not found: some_tool`

    **Solutions**:
    ```bash
    # Check if tool is installed
    which some_tool

    # Install with homebrew
    brew install some_tool

    # Check your PATH
    echo $PATH

    # Restart terminal after installation
    ```
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Git Authentication Issues

    **Problem**: `Permission denied (publickey)` or `Authentication failed`

    **Solutions**:
    ```bash
    # Authenticate with GitHub CLI
    gh auth login

    # Check authentication status
    gh auth status

    # Use HTTPS instead of SSH for cloning
    gh repo clone username/repo
    ```
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### uv and Python Issues

    **Problem**: Package installation failures or version conflicts

    **Solutions**:
    ```bash
    # Update uv to latest version
    brew upgrade uv

    # Clear uv cache
    uv cache clean

    # Check uv version
    uv --version

    # Run with verbose output for debugging
    uv run --verbose python script.py
    ```
    """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""## Terminal Customization (Optional)""")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Level Up Your Terminal Experience

    **iTerm2**: Advanced terminal replacement
    - Split panes, better search, customization
    - Download: https://iterm2.com/

    **Starship Prompt**: Beautiful, informative prompt
    ```bash
    brew install starship
    echo 'eval "$(starship init zsh)"' >> ~/.zshrc
    ```

    **Warp**: Modern terminal with AI features
    - GPU-accelerated, collaborative features
    - Download: https://www.warp.dev/

    **Color Schemes**: Make it beautiful
    - Catppuccin: https://github.com/catppuccin
    - Apply to terminal, VS Code, everything!
    """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""## Next Steps""")
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Practice Exercises

    Try these in your terminal:

    1. **File Management**:
       ```bash
       mkdir practice_project
       cd practice_project
       touch analysis.py requirements.txt README.md
       ls -la
       ```

    2. **Git Workflow**:
       ```bash
       git init
       git add .
       git commit -m "Initial project setup"
       git status
       ```

    3. **Python with uv**:
       ```bash
       echo "print('Hello from uv!')" > hello.py
       uv run python hello.py
       ```

    ### What You've Learned

    ✅ Terminal navigation and file operations  
    ✅ Git version control workflow  
    ✅ Python environment management with uv  
    ✅ Running marimo notebooks professionally  
    ✅ Troubleshooting common issues  

    **You're now ready for professional data science workflows!** 🚀
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Resources for Continued Learning

    - **Git Documentation**: https://git-scm.com/doc
    - **uv Documentation**: https://docs.astral.sh/uv/
    - **Marimo Documentation**: https://docs.marimo.io/
    - **Terminal Cheat Sheet**: https://github.com/0nn0/terminal-mac-cheatsheet
    - **Homebrew**: https://brew.sh/

    **Remember**: You don't need to memorize everything. Focus on the workflow, and look up syntax as needed!
    """
    )
    return


if __name__ == "__main__":
    app.run()
