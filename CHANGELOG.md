# Changelog

## [1.5.1] - 2024-12-20

### Fixed
- Fixed 'show' command to display exactly the requested number of commits by using git's native max_count parameter
- Removed unnecessary filtering logic from basic commit display

## [1.5.0] - 2024-12-20

### Added
- Advanced commit filtering capabilities:
  - Filter by file/directory path using --path option
  - Filter by commit type (merge/regular) using --commit-type option
  - Search commits by message content using --message option
  - Filter by commit size using --size option (small/medium/large/very_large)
- Improved commit size categorization
- Enhanced path-based commit filtering with support for renamed files

## [1.4.0] - 2024-12-20

### Added
- Most modified files analysis functionality
- New command-line option: --modified to display file modification statistics
- Per-file metrics including number of changes, insertions, and deletions
- Color-coded output for better readability

## [1.3.0] - 2024-12-20

### Added
- Branch lifetime analysis functionality
- New command-line option: --lifetime to display branch analysis
- Per-branch metrics including creation date, last activity, and commit frequency
- Color-coded branch information for better readability

## [1.2.0] - 2024-12-20

### Added
- Author contribution statistics with detailed metrics
- New command-line option: --stats to display author statistics
- Statistics include: commits, insertions, deletions, and files changed
- Percentage-based contribution analysis

## [1.1.0] - 2024-12-20

### Added
- Commit frequency heatmap visualization
- Added new dependencies: numpy and termcolor
- New command-line option: --heatmap/-h
- Code churn metrics (lines added/removed) for each commit

## [1.0.0] - 2024-12-20

### Added
- Initial release of Git Repository Visualizer
- Core functionality for visualizing git commit history
- Branch visualization with color coding
- Merge point indication
- Filtering capabilities by author and date range
- Cross-platform support with colorama
- Command-line interface with click
- Documentation in README.md

## [Unreleased]

### Added
- Lazy loading support for large repositories
  - Added pagination with configurable chunk size
  - New CLI options: `--chunk-size` and `--page`
  - Improved memory efficiency for large repositories
- Parallel processing support for commit statistics calculation using multiprocess library
- Added multiprocess>=0.70.15 dependency for parallel processing capabilities
- Memory-efficient handling of large histories
  - Implemented commit stats caching with size limits
  - Added streaming commit processing with batch operations
  - Optimized branch structure handling using generators and sets
  - Added max_cache_size parameter to control memory usage
- Interactive mode using prompt_toolkit
  - New command-line flag `-i` or `--interactive` to start in interactive mode
  - Commands: show, filter, branches, stats, clear, help, quit
  - Command auto-completion
  - Colored output with custom styling
  - Filter persistence between commands
  - Error handling for invalid inputs

### Dependencies
- Added prompt_toolkit>=3.0.43 for interactive mode support
