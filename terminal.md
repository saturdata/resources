Currently, season 1 resources are organized into a folder named for a specific topic such as `data-transformation` with an `overview.md` containing bullet points for that topic and a `resource.py` marimo notebook showing examples on that topic.

Create a folder called terminal containg info on these topics: bullets in `overview.md` and a `resource.py` marimo notebook

- Mac/Unix/Linux terminal and Git basics for data scientists, analysts, and engineers
    - You'll need this to open and run the marimo notebooks we use as resources!
    - Nobody really learns this "correctly" or remembers how they learned this, so we figured we'd give it a try
    - Open terminal (Cmd+Space to spotlight search, then open a terminal)
    - Why learn terminal? You need it to test code like an engineer, work with CLI (command-line interface tools) for data work, and it makes a lot of things so. much. faster by avoiding mouse clicks
        - Emphasize the UI equivalent of each command (i.e. `rm` is like clicking delete in finder to delete a file)
        - Navigation: ls, pwd, cd into a dir and cd back with `cd ..`
        - Functional commands: open, mkdir, mv, cp, rm, touch or vi
        - Run a Python file: `python file.py` (or `python3 file.py`) like an engineer
        - You can partially type paths and file named, and hit tab to use terminal auto-complete as well
        - File editing with vim: vi opens a basic file editor called vim, i to insert and enter edit mode, esc to leave edit mode and return to view only mode, colon q to quit without editing, colon wq to write and quit. ls to check that files were created
    - Storing data work: I've written SQL in Excel cells and stored "production" code in Microsoft Teams folders. Spoilers, it doesn't work
        - This problem is solved by a version control system called Git using a system called GitHub, where you track changes to files with a full history so you can always go back. You've probably used similar features in Google drive
        - Install git with homebew: `brew install git` and install the GitHub CLI: `brew install gh`
        - To use code from an existing repository (aka repo) on GitHub, you need to clone that repo locally to your computer. The easiest way is with the GitHub CLI. To clone our resources repo, run: `gh repo clone saturdata/resources`
        - When you make a change to your code, you need to make that change at three levels: stage or add the change, commit the change to your repository with a descriptive message, and push the change
        - Commands to execute this: `git add .` (or git add all changes, or `git add filename.extension` to add only a specific change), `git commit -m "your commit message here"` (note that the `m` here is called a flag), and `git push` fropm your local clone to the cloud repository that we consider the source of truth
    - venvs and dependencies
        - The tools and Python packages/libraries (like pandas and Polars for data work) that are required to run your code are called dependencies.
        - They may differ across projects, or worse, one package may break another if the version is too old or too new. This is called a dependency conflict
        - The way we solve for this is with something called a virtual environment (env), commonly shortened to venv
        - This way, you can work on projects that have different dependency requirements, without affecting each other
        - There are a thousand ways to create these venvs. Once you have one, you would activate it with a command like this: `source <venv_name>/bin/activate`. This command is actually running a program that will activate your venv so you have access to all the tools you need in one place
        - One package and venv manager has emerged to rule them all: uv. Install with `brew install uv`
        - UV is awesome because it's lightning fast and allows you to essentially dynamically create a virtual environment when you run a file
        - Marimo is a modern framework for Python and SQL notebooks similar to Jupyter but with a lot of super cool engineering features that work closely with UV, as you'll see below
    - Running one of our marimo notebooks
        - Now it's time to see all of this pay off!
        - Assuming you cloned our repo above (`gh repo clone saturdata/resources`) and installed uv (`brew install uv`), you're ready to run one of our resource notebooks
        - cd into the directory where you have our repo cloned: `cd resources`
        - Now run this command, and you'll see our resource notebook pop up in the browser: `uv run marimo run season-1/data-visualization/resource.py`
        - Let's break down what the command is doing: The `uv run` part is handling all the dependencies for us and creating a venv behind the scenes, so we don't have to worry about dependencies at all
        - The `marimo run` part is allowing us to host the Marimo web application on our local machine
        - The `season-1/data-visualization/resource.py` part is just a path specifying that we want to look in the season 1 folder, the data viz subfolder, and specifically open the resources notebook.
            - Alternatively, you could run `cd season-1/data-visualization` first and then use a relative path: `uv run marimo run resource.py`
    - Conclusion: You don't have to be a terminal whiz, but you do need to know the basics in order to check out our resources. Happy building!

- End: Can customize your terminal with iTerm2, Starship(https://starship.rs/)/Catpuccin, and Warp for vibe coding 