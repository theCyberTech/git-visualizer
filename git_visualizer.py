#!/usr/bin/env python3

import sys
import os
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Tuple
from collections import defaultdict
import click
from git import Repo, GitCommandError
from git.objects import Commit
import colorama
from colorama import Fore, Style
import numpy as np
from termcolor import colored
import multiprocess as mp
from functools import partial
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.shortcuts import clear
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.styles import Style
import itertools

# Initialize colorama for cross-platform color support
colorama.init()

class GitVisualizer:
    # Size thresholds for commit categorization (per operation type)
    TINY_CHANGE_THRESHOLD = 3
    SMALL_CHANGE_THRESHOLD = 10
    MEDIUM_CHANGE_THRESHOLD = 50

    def __init__(self, repo_path: str, chunk_size: int = 100, max_cache_size: int = 1000):
        """Initialize GitVisualizer with repository path."""
        self.repo_path = os.path.abspath(repo_path)
        self._validate_repo()
        self.repo = Repo(self.repo_path)
        self.branch_colors = {
            'main': Fore.GREEN,
            'master': Fore.GREEN,
        }
        self.terminal_width = os.get_terminal_size().columns
        self.chunk_size = chunk_size
        self.max_cache_size = max_cache_size
        self._stats_cache = {}

    def _validate_repo(self) -> None:
        """Validate if the given path is a valid git repository."""
        if not os.path.exists(self.repo_path):
            raise ValueError(f"Repository path does not exist: {self.repo_path}")
        if not os.path.exists(os.path.join(self.repo_path, '.git')):
            raise ValueError(f"Not a valid git repository: {self.repo_path}")

    def _get_branch_color(self, branch_name: str) -> str:
        """Get color for branch, creating new ones for unknown branches."""
        if branch_name not in self.branch_colors:
            colors = [Fore.BLUE, Fore.YELLOW, Fore.MAGENTA, Fore.CYAN]
            self.branch_colors[branch_name] = colors[len(self.branch_colors) % len(colors)]
        return self.branch_colors[branch_name]

    def _get_commit_stats_parallel(self, commits: List[Commit]) -> List[Tuple[int, int]]:
        """Get commit statistics in parallel for a list of commits."""
        # Filter out cached commits
        uncached_commits = [c for c in commits if c.hexsha not in self._stats_cache]
        
        # Get stats for uncached commits
        if uncached_commits:
            with mp.Pool() as pool:
                new_stats = pool.map(self._get_commit_stats, uncached_commits)
                
            # Update cache with new stats
            for commit, stats in zip(uncached_commits, new_stats):
                self._stats_cache[commit.hexsha] = stats
                
            # Maintain cache size limit
            if len(self._stats_cache) > self.max_cache_size:
                # Remove oldest entries
                remove_count = len(self._stats_cache) - self.max_cache_size
                for k in list(self._stats_cache.keys())[:remove_count]:
                    del self._stats_cache[k]
        
        # Return stats for all commits (from cache)
        return [self._stats_cache[c.hexsha] for c in commits]

    def _get_commit_stats(self, commit: Commit) -> Tuple[int, int]:
        """Get the number of lines added and removed in a commit."""
        try:
            if len(commit.parents) > 0:
                parent = commit.parents[0]
                diff = parent.diff(commit, create_patch=True)
                lines_added = sum(d.diff.count(b'\n+') - 1 if d.diff else 0 for d in diff)
                lines_removed = sum(d.diff.count(b'\n-') - 1 if d.diff else 0 for d in diff)
                return lines_added, lines_removed
            else:  # First commit
                diff = commit.diff(None, create_patch=True)
                lines_added = sum(d.diff.count(b'\n+') - 1 if d.diff else 0 for d in diff)
                return lines_added, 0
        except GitCommandError:
            return 0, 0

    def _format_commit(self, commit: Commit, branch_name: str) -> str:
        """Format commit information with color and proper width."""
        short_hash = commit.hexsha[:7]
        author = commit.author.name
        date = commit.authored_datetime.strftime('%Y-%m-%d %H:%M')
        message = commit.message.split('\n')[0]
        
        # Get code churn metrics
        lines_added, lines_removed = self._get_commit_stats(commit)
        churn_stats = f"+{lines_added}/-{lines_removed}"

        # Format with color based on branch
        color = self._get_branch_color(branch_name)
        formatted = f"{color}[{branch_name}] {short_hash} {author} {date} ({churn_stats}){Fore.RESET}"
        
        # Ensure the message fits in the terminal
        max_msg_width = self.terminal_width - len(formatted) - 5
        if len(message) > max_msg_width:
            message = message[:max_msg_width-3] + "..."
        
        return f"{formatted} {message}"

    def _is_merge_commit(self, commit: Commit) -> bool:
        """Check if a commit is a merge commit."""
        return len(commit.parents) > 1

    def _commit_affects_path(self, commit: Commit, path: str) -> bool:
        """Check if a commit affects a specific file or directory path."""
        try:
            if len(commit.parents) > 0:
                parent = commit.parents[0]
                diffs = parent.diff(commit)
                for diff in diffs:
                    if path in (diff.a_path or '') or path in (diff.b_path or ''):
                        return True
            else:  # First commit
                diffs = commit.diff(None)
                for diff in diffs:
                    if path in (diff.b_path or ''):
                        return True
            return False
        except GitCommandError:
            return False

    def _get_commit_size_category(self, lines_added: int, lines_removed: int) -> str:
        """Categorize commit based on line changes."""
        if lines_added <= 3 and lines_removed <= 2:
            return "small"  # Includes (+0/-0), (+3/-2)
        else:
            return "large"  # (+10/-4) etc

    def visualize(
        self,
        date_range: Optional[tuple] = None,
        author: Optional[str] = None,
        path_filter: Optional[str] = None,
        commit_type: Optional[str] = None,
        message_filter: Optional[str] = None,
        size_filter: Optional[str] = None,
        start_index: int = 0,
        limit: Optional[int] = None
    ) -> Tuple[List[Commit], bool]:
        """
        Visualize the git repository with optional filters and pagination.
        Memory-efficient implementation using generators and streaming.
        """
        def commit_matches_filters(commit: Commit) -> bool:
            """Helper function to check if commit matches all filters."""
            if date_range and not (date_range[0] <= commit.authored_datetime <= date_range[1]):
                return False
                
            if author and author.lower() not in commit.author.name.lower():
                return False
                
            if path_filter and not self._commit_affects_path(commit, path_filter):
                return False
                
            if commit_type:
                is_merge = commit_type.lower() == 'merge'
                if self._is_merge_commit(commit) != is_merge:
                    return False
                    
            if message_filter and message_filter.lower() not in commit.message.lower():
                return False
                
            if size_filter:
                lines_added, lines_removed = self._get_commit_stats(commit)
                if self._get_commit_size_category(lines_added, lines_removed) != size_filter:
                    return False
                    
            return True

        try:
            # Stream commits instead of loading all at once
            commits_iter = self.repo.iter_commits('--all', skip=start_index, max_count=limit)
            filtered_commits = []
            
            # Get exactly the number of commits requested
            for commit in commits_iter:
                filtered_commits.append(commit)

            if not filtered_commits:
                click.echo("No commits found.")
                return [], False

            # Check if there are more commits
            has_more = bool(next(self.repo.iter_commits('--all', skip=start_index + len(filtered_commits), max_count=1), None))

            # Build branch structure efficiently using generators
            branches = {}
            for branch in self.repo.heads:
                branches[branch.name] = set(c.hexsha for c in self.repo.iter_commits(branch.name, max_count=len(filtered_commits)))

            # Print commit history
            for commit in filtered_commits:
                containing_branches = [
                    name for name, branch_commits in branches.items()
                    if commit.hexsha in branch_commits
                ]
                
                if not containing_branches:
                    continue

                primary_branch = containing_branches[0]
                print(self._format_commit(commit, primary_branch))

                if len(commit.parents) > 1:
                    merge_branches = [b for b in containing_branches if b != primary_branch]
                    if merge_branches:
                        indent = " " * 4
                        print(f"{indent}↳ Merged with: {', '.join(merge_branches)}")

            return filtered_commits, has_more

        except GitCommandError as e:
            click.echo(f"Git error: {str(e)}", err=True)
            sys.exit(1)
        except Exception as e:
            click.echo(f"Error: {str(e)}", err=True)
            return [], False

    def generate_heatmap(self) -> None:
        """Generate and display a commit frequency heatmap."""
        # Get all commits
        commits = list(self.repo.iter_commits('--all'))
        if not commits:
            click.echo("No commits found in the repository.")
            return

        # Create a dictionary to store commit counts by date
        commit_counts = defaultdict(int)
        
        # Get date range
        newest_date = max(c.authored_datetime.date() for c in commits)
        oldest_date = min(c.authored_datetime.date() for c in commits)
        
        # Count commits by date
        for commit in commits:
            commit_date = commit.authored_datetime.date()
            commit_counts[commit_date] += 1

        # Calculate the maximum commits in a day for scaling
        max_commits = max(commit_counts.values()) if commit_counts else 1
        
        # Create week-based matrix for the heatmap
        weeks = (newest_date - oldest_date).days // 7 + 1
        heatmap = np.zeros((7, weeks))  # 7 rows for days of week
        
        # Fill the heatmap matrix
        current_date = oldest_date
        while current_date <= newest_date:
            week_idx = (current_date - oldest_date).days // 7
            day_idx = current_date.weekday()
            count = commit_counts.get(current_date, 0)
            heatmap[day_idx][week_idx] = count
            current_date += timedelta(days=1)

        # Display the heatmap
        days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        
        click.echo("\nCommit Frequency Heatmap:")
        click.echo(f"Period: {oldest_date} to {newest_date}")
        click.echo(f"Total commits: {len(commits)}")
        click.echo("")

        # Print the heatmap
        for day_idx, day in enumerate(days):
            sys.stdout.write(f"{day} ")
            for week in range(weeks):
                count = int(heatmap[day_idx][week])
                if count == 0:
                    color = 'white'
                elif count < max_commits * 0.25:
                    color = 'green'
                elif count < max_commits * 0.5:
                    color = 'yellow'
                elif count < max_commits * 0.75:
                    color = 'red'
                else:
                    color = 'red'
                
                intensity = '⬤' if count > 0 else '◯'
                sys.stdout.write(colored(intensity, color))
            sys.stdout.write('\n')
        
        # Print legend
        click.echo("\nLegend:")
        click.echo(f"{colored('◯', 'white')} 0 commits  ", nl=False)
        click.echo(f"{colored('⬤', 'green')} 1-{int(max_commits * 0.25)} commits  ", nl=False)
        click.echo(f"{colored('⬤', 'yellow')} {int(max_commits * 0.25 + 1)}-{int(max_commits * 0.5)} commits  ", nl=False)
        click.echo(f"{colored('⬤', 'red')} {int(max_commits * 0.5 + 1)}+ commits")

    def get_author_statistics(self) -> Dict[str, Dict[str, int]]:
        """
        Get comprehensive statistics about author contributions.
        
        Returns:
            Dict containing author statistics with the following metrics:
            - commits: number of commits
            - insertions: number of lines added
            - deletions: number of lines deleted
            - files_changed: number of files modified
        """
        author_stats = defaultdict(lambda: {
            'commits': 0,
            'insertions': 0,
            'deletions': 0,
            'files_changed': set()
        })
        
        try:
            for commit in self.repo.iter_commits('--all'):
                author_name = commit.author.name
                author_stats[author_name]['commits'] += 1
                
                try:
                    # Get the stats for this commit
                    if len(commit.parents) > 0:
                        diff_index = commit.parents[0].diff(commit)
                        stats = commit.stats.total
                        author_stats[author_name]['insertions'] += stats['insertions']
                        author_stats[author_name]['deletions'] += stats['deletions']
                        # Track unique files changed
                        for diff in diff_index:
                            if diff.a_path:
                                author_stats[author_name]['files_changed'].add(diff.a_path)
                            if diff.b_path:
                                author_stats[author_name]['files_changed'].add(diff.b_path)
                except (GitCommandError, KeyError) as e:
                    print(f"Warning: Could not get stats for commit {commit.hexsha[:7]}: {str(e)}")
                    continue
                    
        except GitCommandError as e:
            print(f"Error getting repository statistics: {str(e)}")
            return {}
        
        # Convert sets to counts for JSON serialization
        for author in author_stats:
            author_stats[author]['files_changed'] = len(author_stats[author]['files_changed'])
            
        return dict(author_stats)

    def print_author_statistics(self) -> None:
        """Display formatted author contribution statistics."""
        stats = self.get_author_statistics()
        if not stats:
            print("No author statistics available.")
            return
            
        # Calculate totals for percentage calculations
        total_commits = sum(author['commits'] for author in stats.values())
        total_insertions = sum(author['insertions'] for author in stats.values())
        total_deletions = sum(author['deletions'] for author in stats.values())
        total_files = max(author['files_changed'] for author in stats.values())
        
        # Print header
        print("\nAuthor Contribution Statistics:")
        print("-" * self.terminal_width)
        header = f"{'Author':<30} {'Commits':<15} {'Insertions':<15} {'Deletions':<15} {'Files Changed':<15}"
        print(header)
        print("-" * self.terminal_width)
        
        # Print each author's stats
        for author, data in sorted(stats.items(), key=lambda x: x[1]['commits'], reverse=True):
            commit_pct = (data['commits'] / total_commits * 100) if total_commits > 0 else 0
            insertion_pct = (data['insertions'] / total_insertions * 100) if total_insertions > 0 else 0
            deletion_pct = (data['deletions'] / total_deletions * 100) if total_deletions > 0 else 0
            
            author_line = (
                f"{author[:29]:<30} "
                f"{data['commits']:>6} ({commit_pct:>6.1f}%) "
                f"{data['insertions']:>6} ({insertion_pct:>6.1f}%) "
                f"{data['deletions']:>6} ({deletion_pct:>6.1f}%) "
                f"{data['files_changed']:>6}"
            )
            print(author_line)
        
        print("-" * self.terminal_width)

    def analyze_branch_lifetime(self) -> None:
        """Analyze and display branch lifetimes and activity patterns."""
        branch_data = defaultdict(lambda: {'created': None, 'last_commit': None, 'commit_count': 0})
        
        # Analyze all branches
        for branch in self.repo.heads:
            commits = list(self.repo.iter_commits(branch))
            if not commits:
                continue
                
            branch_name = branch.name
            first_commit = commits[-1]
            last_commit = commits[0]
            
            branch_data[branch_name].update({
                'created': first_commit.authored_datetime,
                'last_commit': last_commit.authored_datetime,
                'commit_count': len(commits)
            })
        
        # Display results
        click.echo("\nBranch Lifetime Analysis:")
        click.echo("-" * self.terminal_width)
        
        for branch_name, data in branch_data.items():
            if not data['created']:
                continue
                
            color = self._get_branch_color(branch_name)
            lifetime = data['last_commit'] - data['created']
            days = lifetime.days
            
            # Format the output
            created_str = data['created'].strftime('%Y-%m-%d')
            last_commit_str = data['last_commit'].strftime('%Y-%m-%d')
            
            click.echo(f"{color}Branch: {branch_name}{Fore.RESET}")
            click.echo(f"  Created: {created_str}")
            click.echo(f"  Last Activity: {last_commit_str}")
            click.echo(f"  Lifetime: {days} days")
            click.echo(f"  Total Commits: {data['commit_count']}")
            click.echo(f"  Activity Rate: {data['commit_count']/max(1, days):.2f} commits/day")
            click.echo("-" * self.terminal_width)

    def _get_most_modified_files(self, commits: List[Commit]) -> Dict[str, Dict[str, int]]:
        """Analyze and return statistics for the most modified files."""
        file_stats = defaultdict(lambda: {'changes': 0, 'insertions': 0, 'deletions': 0})
        
        for commit in commits:
            if len(commit.parents) > 0:
                parent = commit.parents[0]
                diffs = parent.diff(commit, create_patch=True)
                for diff in diffs:
                    if diff.a_path:
                        path = diff.a_path
                        file_stats[path]['changes'] += 1
                        if diff.diff:
                            file_stats[path]['insertions'] += diff.diff.count(b'\n+') - 1
                            file_stats[path]['deletions'] += diff.diff.count(b'\n-') - 1
        
        return dict(file_stats)

    def display_most_modified_files(self, top_n: int = 10) -> None:
        """Display statistics for the most modified files."""
        commits = list(self.repo.iter_commits('--all'))
        file_stats = self._get_most_modified_files(commits)
        
        # Sort files by total changes
        sorted_files = sorted(
            file_stats.items(),
            key=lambda x: x[1]['changes'],
            reverse=True
        )[:top_n]
        
        click.echo("\nMost Modified Files:")
        click.echo("-" * self.terminal_width)
        
        for file_path, stats in sorted_files:
            changes = stats['changes']
            insertions = stats['insertions']
            deletions = stats['deletions']
            
            # Format the statistics with color
            stats_str = (
                f"{Fore.CYAN}{file_path}{Fore.RESET} - "
                f"Changes: {Fore.YELLOW}{changes}{Fore.RESET}, "
                f"(+{Fore.GREEN}{insertions}{Fore.RESET}/"
                f"-{Fore.RED}{deletions}{Fore.RESET})"
            )
            click.echo(stats_str)

    def interactive_mode(self):
        """Start an interactive session for exploring the repository."""
        session = PromptSession()
        commands = WordCompleter(['help', 'show', 'filter', 'clear', 'quit', 'branches', 'stats'])
        
        style = Style.from_dict({
            'command': '#ansicyan',
            'arg': '#ansigreen',
            'error': '#ansired',
            'help': '#ansiyellow',
        })

        help_text = """
Available commands:
  show [n]              - Show n commits (default: 10)
  filter <options>      - Filter commits (author/since/until/path/type/message)
  branches             - List all branches
  stats               - Show repository statistics
  clear               - Clear the screen
  help                - Show this help message
  quit                - Exit interactive mode
        """

        print("Welcome to Git Visualizer Interactive Mode! Type 'help' for available commands.")
        
        current_filter = {
            'date_range': None,
            'author': None,
            'path_filter': None,
            'commit_type': None,
            'message_filter': None,
            'size_filter': None
        }
        
        while True:
            try:
                command = session.prompt('git-viz> ', completer=commands, style=style)
                args = command.strip().split()
                
                if not args:
                    continue
                    
                if args[0] == 'quit':
                    break
                    
                elif args[0] == 'help':
                    print(help_text)
                    
                elif args[0] == 'clear':
                    clear()
                    
                elif args[0] == 'show':
                    size = int(args[1]) if len(args) > 1 else 10
                    # Just get the raw commits without any filtering
                    commits = list(self.repo.iter_commits('--all', max_count=size))
                    for commit in commits:
                        print(self._format_commit(commit, self.repo.active_branch.name))
                    if len(list(self.repo.iter_commits('--all', max_count=1, skip=size))) > 0:
                        print("\nMore commits available. Use 'show' with a larger number to view more.")
                    
                elif args[0] == 'filter':
                    if len(args) < 3:
                        print("Usage: filter <option> <value>")
                        print("Options: author, since, until, path, type, message, size")
                        continue
                        
                    option, value = args[1], ' '.join(args[2:])
                    if option == 'author':
                        current_filter['author'] = value
                    elif option in ['since', 'until']:
                        try:
                            date = datetime.strptime(value, '%Y-%m-%d')
                            if current_filter['date_range'] is None:
                                current_filter['date_range'] = [datetime.min, datetime.max]
                            if option == 'since':
                                current_filter['date_range'][0] = date
                            else:
                                current_filter['date_range'][1] = date
                        except ValueError:
                            print("Date must be in YYYY-MM-DD format")
                            continue
                    elif option == 'path':
                        current_filter['path_filter'] = value
                    elif option == 'type':
                        current_filter['commit_type'] = value
                    elif option == 'message':
                        current_filter['message_filter'] = value
                    elif option == 'size':
                        if value in ['small', 'large']:
                            current_filter['size_filter'] = value
                        else:
                            print("Size must be one of: small, large")
                            continue
                    else:
                        print(f"Unknown filter option: {option}")
                        continue
                    print(f"Filter updated: {option}={value}")
                        
                elif args[0] == 'branches':
                    for branch in self.repo.heads:
                        print(f"{self._get_branch_color(branch.name)}{branch.name}{Fore.RESET}")
                        
                elif args[0] == 'stats':
                    total_commits = len(list(self.repo.iter_commits()))
                    total_branches = len(list(self.repo.heads))
                    print(f"\nRepository Statistics:")
                    print(f"Total commits: {total_commits}")
                    print(f"Total branches: {total_branches}")
                    print(f"Active branch: {self.repo.active_branch.name}")
                    
                else:
                    print(f"Unknown command: {args[0]}")
                    
            except KeyboardInterrupt:
                continue
            except EOFError:
                break
            except Exception as e:
                print(f"Error: {str(e)}")

@click.command()
@click.argument('repo_path', type=click.Path(exists=True))
@click.option('--author', help='Filter commits by author name')
@click.option('--since', help='Show commits more recent than specified date')
@click.option('--until', help='Show commits older than specified date')
@click.option('--path', help='Filter commits that modify given path')
@click.option('--commit-type', type=click.Choice(['merge', 'regular']), help='Filter by commit type')
@click.option('--message', help='Filter commits by message content')
@click.option('--size', type=click.Choice(['small', 'large']), help='Filter by number of line changes')
@click.option('--chunk-size', type=int, default=100, help='Number of commits to load at once')
@click.option('--page', type=int, default=1, help='Page number for pagination')
@click.option('--interactive', '-i', is_flag=True, help='Start in interactive mode')
def main(repo_path, author, since, until, path, commit_type, message, size, chunk_size, page, interactive):
    """Visualize a git repository's commit history with pagination support."""
    try:
        visualizer = GitVisualizer(repo_path, chunk_size=chunk_size)
        
        if interactive:
            visualizer.interactive_mode()
        else:
            date_range = None
            if since or until:
                start_date = datetime.strptime(since, '%Y-%m-%d') if since else datetime.min
                end_date = datetime.strptime(until, '%Y-%m-%d') if until else datetime.max
                date_range = (start_date, end_date)

            start_index = (page - 1) * chunk_size
            commits, has_more = visualizer.visualize(
                date_range=date_range,
                author=author,
                path_filter=path,
                commit_type=commit_type,
                message_filter=message,
                size_filter=size,
                start_index=start_index
            )
            
            if has_more:
                click.echo(f"\nMore commits available. Use --page {page + 1} to view next page.")

    except ValueError as e:
        click.echo(f"Error: {str(e)}", err=True)
        sys.exit(1)

if __name__ == '__main__':
    main()
