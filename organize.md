I want to logically split up analytics_basics.py into 2-3 separate Marimo notebooks. Each one should be a unified focus area of analytics.

These notebooks will be used as resources for a podcast. Each podcast episode will have 2 parts: a 20-30 minute discussion introducting the topics at hand, and then a 30-45 minute part going through the notebook resource we provide.

Split up the notebook keeping in mind this structure: Product subfolders within the season-1 dir. Each subfolder should be the topic name (with hyphens instead of spaces) and contain two files. The first file is a markdown document called overview.md containig bullet points for discussion of the topics in the marimo notebook. The second file should be resource.py with the marimo resources. Feel free to add a few cells at the beginning of each marimo notebook to introduce it.

Example folder inside season-1 dir:

name: data-transformation
files:
overview.md
- numpy
- pandas
- polars
- comparison of pandas and polars operations
- comparison to sql
- which kinds of data operations do people need to get very comfortable with, and which ones aren't as important

resource.py
Parts of the analytics_basics.py notebook relevant to the above
