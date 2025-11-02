# Terminal and Git: The Professional Workflow

## The Moment You Realized You Were Doing It Wrong

Remember when you spent 20 minutes clicking through folders to find that data file, and then watched your engineering colleague navigate there in 3 seconds using terminal commands? That was the moment you realized you were working like an amateur, not like a data professional.

## Why Terminal Matters: The Gateway to Professional Work

**The Workplace Reality**: Every data tool worth using has a CLI (command-line interface) component. Jupyter? Terminal. Git? Terminal. Python environments? Terminal. The moment you accept that terminal isn't optional—it's fundamental—is the moment you stop fighting your tools and start using them.

**The Misconception**: "I can do everything in a GUI, so why learn terminal?"

**The Hard Truth**: GUIs are great for exploration. Terminals are great for speed, automation, and professional workflows. That script you need to run every day? GUI: click, wait, click, wait. Terminal: one command, automated. The difference is time **and** scalability. You can't automate GUI clicks. You can automate terminal commands.

**The 'Aha' Moment**: Terminal commands are faster and **reproducible**. That sequence of clicks you did to set up your environment can't really be documented, but a sequence of terminal commands can be copy-pasted. Reproducibility is the foundation of professional data work, and terminal commands are reproducible by design.

**Testing Code Like an Engineer**: Data professionals who can't test their code in terminal are like chefs who can't taste their food. The terminal is where you verify things work before they go to production. That script that "works in Jupyter" but fails when you actually run it might have been caught by a terminal check.

## Core Navigation: The Foundation

**`pwd` - Know Where You Are**: If you're lost in nested folders and realize you have no idea where you are, `pwd` saves you. It's your compass command. Every other navigation command is relative to where you are, so knowing where you are is step one.

**`ls` - See What's Available**: You can't navigate what you can't see. `ls` shows you the files and folders. `ls -la` shows you everything including hidden files (those dot-files that start with `.`). Permissions, ownership, sizes, dates—everything you need to understand your filesystem.

**`cd` - Moving Around**: Navigation seems trivial until you're trying to explain to someone how to get to `/Users/you/Documents/Projects/data/analysis/results/final/`. That's when you realize relative paths (`cd ..`) and home directory shortcuts (`cd ~`) are crucial.

**The Tab Completion Revelation**: Start typing a filename and press Tab. It completes. This seems minor until you're working with files named `customer_transaction_data_2024_q1_cleaned_final_v2.csv`. Then tab completion becomes a necessity. The muscle memory moment happens when you stop typing full paths and start typing partial paths + Tab.

## File Operations: The Basics That Matter

**`mkdir`, `touch`, `cp`, `mv`, `rm`**: These are your file manipulation vocabulary. The moment you stop thinking "I need to create a folder" and start thinking "I need to `mkdir`" is the moment terminal becomes your native language.

**The `rm -rf` Warning**: That command deletes directories recursively and permanently. It doesn't go to trash. It doesn't ask twice. Terminal gives you power, and with power comes responsibility. The moment you understand that `rm -rf` can't be undone is the moment you become careful. And careful is professional.

**`cat` - View File Contents**: Need to quickly see what's in a file? `cat filename.txt` displays the entire file contents. Perfect for config files, logs, CSV headers, or checking a script before running it. The moment you stop opening files in editors just to read them is the moment you realize `cat` saves time.

**`open` - Your GUI Bridge**: Need to open a file with the default application? `open filename.txt` does it. It's not cheating—it's pragmatic. Terminal for speed, GUI for viewing. The key is knowing when each is appropriate.

## Text Editing: vim Survival Guide

**The Misconception**: "vim is too hard, I'll just use a GUI editor."

**The Reality**: vim is everywhere—on servers, in containers, in remote environments. You can't always use a GUI.

**The Survival Commands**:

- `i` to insert (start typing)
- `Esc` to stop inserting (back to command mode)
- `:wq` to write and quit (save and exit)
- `:q!` to quit without saving (abandon changes)

That's 90% of what you need. The rest you can look up when needed. The breakthrough: Understanding that vim has two modes (insert and normal), and most commands only work in normal mode. Once you get that, vim stops being mysterious.

**The nano Alternative**: If vim feels too hard, `nano filename.txt` is friendlier. Ctrl+X to exit, Y to save. Simple. The key is having *some* terminal editor skills. Whether it's vim or nano doesn't matter—being able to edit files in terminal does.

## Terminal Productivity: The Little Things That Add Up

**Command History with Arrow Keys**: Press ↑ to see previous commands. It seems trivial until you realize you just ran a 50-character command perfectly and don't want to type it again.

**`Ctrl+C` to Cancel**: That command that's taking forever can be stopped by `Ctrl+C`. This seems obvious until you're watching a script hang and realize you don't know how to stop it.

**`&&` for Sequential Commands**: `mkdir project && cd project && touch README.md`—three commands, one line. The breakthrough: Understanding that `&&` means "run the next command only if this one succeeds." It's not just concatenation—it's conditional execution. Combine commands thoughtfully, and terminal becomes a scripting language.

## Git: The Problem It Solves

**The Nightmare Before Git**: `analysis_final.py`, `analysis_final_v2.py`, `analysis_final_ACTUALLY_FINAL.py`. Files stored in Teams folders with no history. No way to see what changed. No way to recover previous versions. Collaboration that creates chaos.

**The Git Solution**: Complete history. Collaboration without conflicts. Revert to any version. Professional development workflows. The moment you understand that Git isn't about code—it's about change management—is the moment it clicks.

**The Misconception**: "Git is too complicated for data work."

**The Reality**: Git is how professionals work. Every technically oriented data team uses version control. Every production pipeline uses version control. Learning Git is fundamental if you want to work professionally.

## Core Git Concepts: The Mental Model

**Repository (repo)**: A project folder tracked by Git. That's it. It's not complicated—it's just "this folder has history."

**Clone**: Download a copy of a remote repository. Not "copy"—clone. Clones have Git history. Copies don't.

**Stage, Commit, Push**: The three-step dance of saving changes:

1. **Stage** (`git add`): "These changes are ready to be saved."
2. **Commit** (`git commit`): "Save these changes with this message."
3. **Push** (`git push`): "Upload my saved changes to the shared repository."

The 'aha' moment?: Understanding that staging is intentional. You don't commit everything—you commit what makes sense together. One logical change = one commit.

**Pull**: Download changes from the remote repository. Always pull before starting new work. Always. The moment you understand that `git pull` is how you stay current is the moment you stop creating merge conflicts.

## Essential Git Workflow: The Daily Routine

**`gh repo clone`**: The GitHub CLI way to clone. It handles authentication. It's simpler than the old way.

**`git status`**: Your diagnostic tool. Don't know what's changed? `git status`. Don't know if you're on the right branch? `git status`. Confused about the repository state? `git status`. Run it frequently. It's like `pwd` for Git—know your state.

**`git add .` vs. `git add filename`**: `git add .` is fast but adds everything. `git add filename` is precise but requires more commands. Both are valid—the choice depends on your workflow.

**Commit Messages That Matter**: "updates" isn't a commit message—it's a cop-out. Good commit messages explain *why* you changed something, not just *what* you changed. "Fix bug in data preprocessing pipeline" beats "updates". "Add visualization for quarterly revenue" beats "fix stuff". Write commits like you're talking to your future self.

**The Pull-Before-Work Rule**: Always `git pull` before starting new work. Always. Yes, even if you just pulled. Yes, even if you're "sure" nothing changed. The moment you make this a habit is the moment you stop dealing with merge conflicts.

## Python Environment Management: The Dependency Hell Solution

**The Nightmare Scenario**: Project A needs pandas 1.5.0. Project B needs pandas 2.0.0. You install pandas 2.0.0 globally. Project A breaks. This isn't hypothetical—it's Tuesday.

**The Misconception**: "I'll just install packages globally and everything will work."

**The Reality**: Different projects need different packages. Different packages need different versions. Global installations create conflicts. Virtual environments solve this by isolating dependencies per project.

**The Traditional Way (venv)**: Create environment, activate it, install packages, deactivate when done. It works, but you have to remember to activate. And deactivate. Every. Single. Time. The mental overhead isn't trivial—it's why people skip it and then wonder why things break.

**The Modern Solution: uv**: Lightning-fast package manager that handles environments automatically. No activation needed. No deactivation needed. Just `uv run python script.py` and it works. The breakthrough? Understanding that uv creates environments behind the scenes automatically. You don't manage them—uv does.

**The `uv run` Magic**: `uv run python script.py` runs your script in an isolated environment with the right dependencies. `uv run marimo run notebook.py` runs your notebook with dependencies handled. You don't think about environments—you just run code. That's the point. Environments should be invisible, not a chore.

## Running Python Code Professionally

**From Worst to Best**:

- `python script.py`: Uses system Python. Can break your system. Don't do this.
- `python3 script.py`: Better, but still not isolated. Risky for projects.
- `uv run python script.py`: Isolated environment, automatic dependency management. Professional.

The moment you switch to `uv run` is the moment Python environments stop being a problem you manage and start being a solution that works.

**Running Marimo Notebooks**: `uv run marimo run notebook.py`. That's it. No activation. No manual environment setup. uv handles everything. The file is a Python file (not JSON like Jupyter), so it's Git-friendly. The workflow is professional. This is how modern data tools should work.

## Troubleshooting: The Real-World Skills

**Permission Denied Errors**: That file you can't execute? `ls -la` shows permissions. `chmod +x filename.py` makes it executable. The moment you understand file permissions is the moment you stop being blocked by "permission denied" errors.

**Command Not Found**: That tool that should be installed but isn't? `which toolname` checks if it exists. `brew install toolname` installs it. The key is knowing your tools: Homebrew for macOS packages, `which` for diagnostics, `echo $PATH` for debugging environment issues.

**Git Authentication Failures**: `gh auth login` solves most GitHub authentication problems. The GitHub CLI handles authentication cleanly. The moment you stop fighting SSH keys and start using `gh auth login` is the moment Git authentication stops being painful.

**uv and Python Issues**: `brew upgrade uv` updates uv. `uv cache clean` clears problematic caches. `uv --version` checks your version. uv is a tool, and like all tools, sometimes it needs maintenance. Knowing the maintenance commands saves hours of debugging.

## Terminal Customization: Leveling Up (Optional)

**iTerm2**: Advanced terminal replacement with split panes, better search, extensive customization. Not required, but if you live in terminal, it's worth it.

**Starship Prompt**: Beautiful prompt that shows Git status, Python environment, and more. Install with Homebrew, add one line to your shell config. Suddenly your terminal becomes informative instead of just functional.

**Warp**: Modern terminal with AI features. GPU-accelerated, collaborative, different paradigm. Worth trying if you're open to new approaches.

**Color Schemes**: Catppuccin and similar schemes make terminal beautiful. Not required, but if you're staring at terminal all day, beautiful beats plain.

Customization isn't about being fancy—it's about making your tools work better for you. If you spend hours in terminal, invest in making it pleasant.

## The Bottom Line

Terminal and Git aren't about memorizing commands—they're about professional workflows. Terminal gives you speed and automation. Git gives you history and collaboration. Together, they're the foundation of how modern data science actually works.

The real skill: Understanding that these tools aren't obstacles—they're amplifiers. Every minute you spend learning terminal and Git pays back in hours saved and frustrations avoided. The moment you stop fighting these tools and start embracing them is the moment you transition from "person who does data work" to "a data professional."

Remember: You don't need to memorize everything. Focus on the workflow. Look up syntax as needed. But understand the patterns: navigation, file operations, version control, environment management. Once you see those patterns, the details become easy.

The terminal isn't scary—it's powerful. Git isn't complicated—it's necessary. And once you're comfortable with both, you're not just writing code anymore. You're working like a professional.
