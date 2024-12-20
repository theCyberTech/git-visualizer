# Git Repository Visualizer

A command-line tool to visualize git repository history with branch visualization and filtering capabilities.

## Features

- Display commit history as an ASCII branch tree
- Show commit hashes, messages, authors, and dates
- Highlight different branches with distinct colors
- Indicate merge points and branch divergences
- Filter commits by date range or author
- Generate commit frequency heatmap
- Cross-platform support (Windows and Unix-like systems)

## Installation

1. Clone this repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Basic usage:
```bash
python git_visualizer.py /path/to/git/repo
```

Interactive mode:
```bash
python git_visualizer.py /path/to/git/repo -i
```

In interactive mode, you can use these commands:
- `show [n]`: Display n most recent commits (default: 10)
- `filter`: Apply various filters to the commit view
- `branches`: List all branches
- `stats`: Show repository statistics
- `help`: Show all available commands
- `quit`: Exit interactive mode

Filter by author:
```bash
python git_visualizer.py /path/to/git/repo --author "John Doe"
```

Filter by date range:
```bash
python git_visualizer.py /path/to/git/repo --since 2024-01-01 --until 2024-12-31
```

Show commit frequency heatmap:
```bash
python git_visualizer.py /path/to/git/repo --heatmap
```

## Options

- `--author`, `-a`: Filter commits by author name
- `--since`, `-s`: Show commits after date (YYYY-MM-DD)
- `--until`, `-u`: Show commits before date (YYYY-MM-DD)
- `--heatmap`, `-h`: Show commit frequency heatmap
- `--interactive`, `-i`: Start in interactive mode

## Requirements

- Python 3.7+
- GitPython
- colorama
- click
- numpy
- termcolor
