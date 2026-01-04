---
name: saturdata-marimo-notebooks
description: Create interactive Marimo notebooks as companion resources to Saturdata podcast episodes. This skill should be used when creating or improving educational data science notebooks for early-career professionals (0-3 years experience). Applies discovery-based learning with progressive complexity. References style.md for educational guidelines and saturdata.md for technical patterns.
---

# Saturdata Marimo Notebooks Skill

## About This Skill

Create interactive Marimo notebooks that serve as companion resources to Saturdata podcast episodes, designed to convert listeners into engaged community members while teaching technical data concepts through hands-on exploration.

**Purpose**: Interactive Marimo notebooks for Saturdata podcast

**Target Audience**: Early-career data professionals (0-3 years experience) transitioning into data engineering and analytics roles

**Learning Philosophy**: Discovery tools, not tutorials. Spark curiosity, build confidence, and create community connections. Users should feel like they learned WITH Shifra and Sam, not FROM them.

**Episode Topics**:
- SQL for data professionals
- Polars/pandas performance comparison
- Data transformation patterns
- Visualization (Seaborn, Plotly, Matplotlib)
- Statistical testing with SciPy
- Terminal and Git fundamentals

**Key Principle**: These notebooks are **discovery tools**, not tutorials. Every notebook should leave users feeling like they learned WITH the hosts, not FROM them.

## When to Use This Skill

Use this skill when:
- Creating new Marimo notebooks for Saturdata podcast episodes
- Improving existing educational notebooks for accuracy or engagement
- Verifying technical accuracy (especially DuckDB SQL syntax)
- Ensuring educational content quality and progressive complexity
- Adding community engagement elements (#Saturdata)
- Validating notebooks follow project standards and conventions

## Core Workflow

Follow this high-level process when working with Saturdata notebooks:

### 1. Understand the Task Type

**New Notebook Creation** � See Task Template: Creating New Notebooks (below)
- Setting up directory structure and scaffolding
- Implementing progressive difficulty levels
- Integrating educational and technical patterns

**Improvement/Accuracy** � See Task Template: Improving Existing Notebooks (below)
- Reading and understanding existing notebook structure
- Verifying technical accuracy with official documentation
- Preserving educational flow and conversational tone

**Data Work** � See saturdata.md "Task Template: Working with Data" (lines 772-857)
- Adding or modifying datasets
- Updating data loading patterns
- Validating data schemas and paths

### 2. Set Up Environment

**Dependencies and Stack** � See saturdata.md "Technical Stack & Setup"
- Python 3.11+ with `uv` package manager
- Core libraries: marimo, polars, pandas, duckdb, numpy, scipy
- Visualization: matplotlib, seaborn, plotly
- Data formats: pyarrow (Parquet), CSV

**Directory Structure** � See saturdata.md "Directory Structure" (lines 53-87)
- `season-1/data/` - Shared datasets for all episodes
- `season-1/{topic}/` - Episode-specific notebooks
- `season-1/{topic}/resource.py` - Main Marimo notebook (consistent naming)

**Path Setup Pattern** � See saturdata.md "File Path Handling" (lines 192-220)
- **Critical**: Always use absolute paths via `Path` module
- Standard pattern: `NOTEBOOK_DIR = Path(__file__).parent if "__file__" in dir() else Path.cwd()`
- Data directory: `DATA_DIR = NOTEBOOK_DIR.parent / "data"`

### 3. Follow Educational Principles

**Learning Philosophy** � See style.md "Core Purpose" (lines 1-12)
- Discovery tools, not tutorials
- Progressive complexity (3 levels: Just Looking, Getting Hands Dirty, Challenge Accepted)
- Can be understood without listening to episode
- 15-30 minutes to complete fully

**Notebook Structure Template** � See style.md "Notebook Structure Template" (lines 6-46)
- Opening hook with episode connection
- 3 progressive exploration levels
- Real-world scenario section
- Discovery bonuses (Easter eggs not in episode)
- Community engagement section

**Content Tone** � See style.md "Language and Tone" (lines 50-68)
- Conversational style ("explaining to a friend")
- Use hosts' voices: "Sam often encounters..." or "Shifra's favorite trick..."
- Avoid jargon without context
- Include humor and relatable frustrations

**Community Engagement** � See style.md "Community Engagement Section" (lines 83-100)
- Include #Saturdata hashtag
- Social sharing prompts
- Challenge problems for community sharing
- LinkedIn/GitHub formatting for solutions

### 4. Implement Technical Patterns

**Marimo Cell Structure** � See saturdata.md "Marimo Notebook Patterns" (lines 88-191)
- Standard cell organization: imports � path setup � data loading � content
- Use `@app.cell` decorator for all cells
- Return values as tuples: `return (variable,)` for single variables
- App configuration: `app = marimo.App(width="medium")`

**Data Loading** � See saturdata.md "Data Handling Standards" (lines 280-348)
- Always use absolute paths with `Path` module
- Convert `Path` to string when passing to libraries: `pl.read_csv(str(path))`
- Validate data after loading (check row counts, required columns)
- Document dataset schema in markdown cells

**DuckDB SQL Integration** � See saturdata.md "DuckDB Integration Pattern" (lines 221-257)
- Use `mo.sql()` for queries (auto-discovers `conn` variable)
- Native DuckDB functions preferred: `median(x)` not `PERCENTILE_CONT`
- Can query Polars/pandas DataFrames directly
- Result has `.value` attribute containing DataFrame

**Visualizations** � See saturdata.md "Visualization Patterns" (lines 499-558)
- Seaborn for statistical/publication-quality plots
- Plotly for interactive web visualizations
- Convert Polars to pandas for Seaborn/Plotly compatibility

## Quick Reference Guide

### Educational Content References

**Tone and Style**:
- Conversational approach � `style.md` lines 50-68
- Progressive difficulty levels � `style.md` lines 13-31
- Community engagement CTAs � `style.md` lines 83-100
- Success metrics and engagement � `style.md` lines 95-100

**Content Structure**:
- Opening hook section � `style.md` lines 6-13
- Concept playground (3 levels) � `style.md` lines 14-31
- Real-world scenarios � `style.md` lines 32-36
- Discovery bonuses � `style.md` lines 37-41
- Final checklist � `style.md` lines 101-111

### Technical Pattern References

**Marimo Notebook Patterns**:
- Standard cell structure � `saturdata.md` lines 88-191
- File path handling (absolute paths) � `saturdata.md` lines 192-220
- DuckDB integration with `mo.sql()` � `saturdata.md` lines 221-257
- Interactive elements and tables � `saturdata.md` lines 258-279

**Data Handling**:
- Dataset requirements � `saturdata.md` lines 280-289
- Shared data assets (transactions, taxi data) � `saturdata.md` lines 290-309
- Data loading patterns with validation � `saturdata.md` lines 310-348

**Code Patterns**:
- Benchmarking function � `saturdata.md` lines 349-414
- Library conversions (Polars �pandass) � `saturdata.md` lines 415-451
- SQL query patterns � `saturdata.md` lines 452-498
- Visualization examples � `saturdata.md` lines 499-558

**Advanced Topics**:
- Performance optimization � `saturdata.md` lines 998-1057
- DuckDB-specific guidelines � `saturdata.md` lines 1058-1130
- DuckDB native functions reference � `saturdata.md` lines 1066-1093

### Common Task References

**Creating New Notebooks**:
- Complete workflow � `saturdata.md` lines 559-715
- Educational structure integration � `style.md` lines 6-46
- Code scaffolding for progressive levels � `saturdata.md` lines 644-708

**Improving Existing Notebooks**:
- Discovery and verification workflow � `saturdata.md` lines 716-771
- DuckDB accuracy verification � `saturdata.md` lines 858-960
- Context7 MCP usage for documentation � `saturdata.md` lines 732-747

**Working with Data**:
- Data discovery and path setup � `saturdata.md` lines 772-857
- Schema documentation patterns � `saturdata.md` lines 808-834
- Validation approaches � `saturdata.md` lines 836-851

## Essential Task Templates

### Creating New Notebooks

High-level workflow for creating a new episode notebook from scratch.

**1. Setup Phase**
```bash
# Navigate to season directory
cd saturdata/resources/season-1/

# Create episode directory (lowercase with hyphens)
mkdir new-topic-name

# Create notebook file (always named resource.py)
touch new-topic-name/resource.py
```

**2. Notebook Scaffolding**

Start with minimal Marimo structure:
- Import cell with `marimo as mo`, `polars as pl`, `duckdb`, `Path`
- Path setup cell with `NOTEBOOK_DIR` and `DATA_DIR`
- Opening markdown cell with episode context and learning objectives
- Progressive content cells (3 levels of difficulty)

**3. Progressive Difficulty Implementation**

Implement three distinct levels:
- **Level 1 "Just Looking"**: Fully implemented code with extensive inline comments
- **Level 2 "Getting Hands Dirty"**: Partial code with TODO sections and hints
- **Level 3 "Challenge Accepted"**: Open-ended problem with multiple valid approaches

**4. Validation Checklist**
- [ ] All cells execute without errors
- [ ] Data loads correctly with absolute paths
- [ ] Visualizations render properly
- [ ] Episode references contextual, not dependent on listening
- [ ] Community engagement CTAs included (#Saturdata)

**Detailed Workflow**: See saturdata.md lines 559-715 for complete step-by-step guide
**Code Scaffolding**: See saturdata.md lines 644-708 for progressive level examples
**Educational Structure**: See style.md lines 6-46 for notebook template

### Improving Existing Notebooks

High-level workflow for enhancing accuracy, fixing bugs, or adding features to existing notebooks.

**1. Discovery Phase**

Read the entire notebook to understand:
- Educational structure and flow
- Episode context and references
- Technical patterns used
- Data dependencies

**2. Verification Using Context7 MCP**

For DuckDB accuracy verification:
```python
# 1. Resolve DuckDB library ID
mcp__context7__resolve-library-id(libraryName="DuckDB")

# 2. Get official documentation
mcp__context7__get-library-docs(
    context7CompatibleLibraryID="/websites/duckdb-stable",
    topic="PIVOT UNPIVOT syntax",  # or specific feature
    mode="code"
)

# 3. Compare notebook syntax with official examples
# 4. Update to native DuckDB patterns when available
```

**3. Pattern Preservation Guidelines**
- Maintain existing educational structure (don't change level progression)
- Preserve conversational tone and host references
- Keep community engagement CTAs
- Respect existing data loading patterns
- Update only technical inaccuracies or add missing features

**4. Quality Assurance Checks**
- [ ] All code executes successfully
- [ ] Technical accuracy verified against official docs
- [ ] Educational flow maintained
- [ ] Conversational tone preserved
- [ ] Data paths still use absolute references
- [ ] No regressions in existing functionality

**Detailed Workflow**: See saturdata.md lines 716-771
**DuckDB Patterns**: See saturdata.md lines 858-960 for common corrections
**Verification Process**: See saturdata.md lines 732-747

### Common Pitfalls to Avoid

**Path Issues**:
- L Relative paths (breaks in different environments)
-  Absolute paths via `Path` module
- See saturdata.md lines 192-220 for correct pattern

**DuckDB SQL Syntax**:
- L PostgreSQL syntax (`PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY x)`)
-  Native DuckDB syntax (`median(x)` or `quantile_cont(x, 0.5)`)
- See saturdata.md lines 858-960 for accuracy guide

**Educational Content**:
- L Assuming users listened to episode
-  Provide full context in notebook
- L Technical jargon without explanation
-  Plain English with conversational tone

**Community Engagement**:
- L Missing #Saturdata hashtag
-  Include social sharing prompts
- L No challenge problems
-  Encourage community sharing with examples

## Quality Checklist Summary

Quick validation checklist with references to detailed validation criteria.

### Code Execution
- [ ] All cells execute without errors in fresh session
- [ ] Absolute paths used for all data loading
- [ ] No dependencies beyond pyproject.toml
- [ ] Type conversions work correctly (especially dates)

**Detailed Checklist**: saturdata.md lines 961-997

### Educational Value
- [ ] Can be understood without listening to episode
- [ ] Progressive difficulty structure (3 levels implemented)
- [ ] Includes at least one "aha moment" not in episode
- [ ] References real workplace scenarios
- [ ] Takes 15-30 minutes to complete fully
- [ ] Conversational tone maintained throughout

**Detailed Checklist**: style.md lines 101-111

### Technical Accuracy
- [ ] Verified against official documentation (use Context7 for DuckDB)
- [ ] Native library functions used where available (e.g., `median()` not `PERCENTILE_CONT`)
- [ ] No deprecated or non-standard syntax
- [ ] Performance patterns follow best practices

**DuckDB Verification**: saturdata.md lines 858-960
**Performance Patterns**: saturdata.md lines 998-1057

### Community Engagement
- [ ] #Saturdata hashtag included
- [ ] Social sharing prompts present
- [ ] LinkedIn/GitHub formatting provided
- [ ] Challenge problems encourage sharing

**Engagement Guidelines**: style.md lines 83-100

### Visual Elements
- [ ] Plots and charts render correctly
- [ ] Interactive tables display properly (use `mo.ui.table()`)
- [ ] Markdown formatting clean and readable
- [ ] Code syntax highlighting works

**Visualization Patterns**: saturdata.md lines 499-558

## References

For detailed technical patterns and educational guidelines, consult:

- **Technical Patterns**: `saturdata.md` - Complete technical reference with code examples
- **Educational Guidelines**: `style.md` - Content design and engagement strategies
- **Skill Creation Guide**: `creating_a_skill.md` - Framework for skill development

**Official Documentation**:
- Marimo: https://docs.marimo.io/
- Polars: https://pola-rs.github.io/polars/
- DuckDB: Use Context7 MCP with `/websites/duckdb-stable` for accuracy
