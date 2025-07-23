"""Git utility functions for the precommit tool."""

import os
import subprocess


def find_git_repositories(path: str, max_depth: int = 3) -> list[str]:
    """Find all git repositories under the given path up to max_depth."""
    repositories = []

    def search_dir(current_path: str, depth: int):
        if depth > max_depth:
            return

        try:
            items = os.listdir(current_path)

            # Check if this is a git repository
            if ".git" in items and os.path.isdir(os.path.join(current_path, ".git")):
                repositories.append(current_path)
                return  # Don't search within git repositories

            # Search subdirectories
            for item in items:
                item_path = os.path.join(current_path, item)
                if os.path.isdir(item_path) and not item.startswith("."):
                    search_dir(item_path, depth + 1)

        except (PermissionError, OSError):
            pass

    search_dir(path, 0)
    return repositories


def run_git_command(repo_path: str, command: list[str]) -> tuple[bool, str]:
    """Run a git command in the specified repository."""
    try:
        result = subprocess.run(["git"] + command, cwd=repo_path, capture_output=True, text=True, check=False)
        return (result.returncode == 0, result.stdout if result.returncode == 0 else result.stderr)
    except Exception as e:
        return (False, str(e))


def get_git_status(repo_path: str) -> dict:
    """Get git status information for a repository."""
    status = {"staged": [], "unstaged": [], "untracked": [], "branch": "unknown"}

    # Get current branch
    success, branch_output = run_git_command(repo_path, ["branch", "--show-current"])
    if success:
        status["branch"] = branch_output.strip()

    # Get status
    success, status_output = run_git_command(repo_path, ["status", "--porcelain"])
    if success:
        for line in status_output.splitlines():
            if not line:
                continue

            status_code = line[:2]
            file_path = line[3:]

            if status_code[0] in "ADMRC":
                status["staged"].append(file_path)
            if status_code[1] in "ADMRC":
                status["unstaged"].append(file_path)
            if status_code == "??":
                status["untracked"].append(file_path)

    return status
