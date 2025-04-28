import logging
import shutil
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

import seutil as su
from jsonargparse import CLI

logger = su.log.get_logger(__name__)


class RefactoringCollector:
    def __init__(
        self,
        work_dir: su.arg.RPath = Path(__file__).parent.parent.parent / "_work",
        download_repo_name: str = "_downloads",
        result_refactoring_repo_name: str = "raw-data",
        raw_repos_name: str = "repos",
        stats_repo_name: str = "stats",
        clone_timeout: int = 300,
        logger_name: str = "get_refactoring.log",
    ):
        self.work_dir = work_dir
        self.download_repo_name = download_repo_name
        self.download_dir = self.work_dir / self.download_repo_name
        self.result_refactoring_repo_name = result_refactoring_repo_name
        self.result_refactoring_dir = self.work_dir / self.result_refactoring_repo_name
        self.stats_repo_name = stats_repo_name
        self.stats_dir = self.work_dir / self.stats_repo_name
        self.CLONE_TIMEOUT = clone_timeout
        self.raw_repos_dir = self.work_dir / raw_repos_name

        logger_path: su.arg.RPath = Path(__file__).parent.parent.parent / logger_name

        su.log.setup(
            logger_path,
            level_stderr=logging.WARNING,  # Console logs WARNING and above
            level_file=logging.INFO,
            fmt_stderr="[{asctime}|{levelname}] {message}",  # New-style formatting with curly braces
            fmt_file="[{asctime}|{levelname}|{lineno}] {message}",
        )

        self.ensure_directories_exist()

    def ensure_directories_exist(self):
        """Create necessary directories if they do not exist."""
        directories = [self.download_dir, self.result_refactoring_dir, self.stats_dir]
        for dir_path in directories:
            if not dir_path.exists():
                dir_path.mkdir(parents=True, exist_ok=True)
                logger.info(f"Created directory: {dir_path}")
            else:
                logger.info(f"Directory already exists: {dir_path}")

    def clone_repository(self, url: str, target_dir: su.arg.RPath, repo_name: str):
        """Clone a single repository into the specified directory using command-line Git."""
        repo_path = target_dir / repo_name
        try:
            if not repo_path.exists():
                logger.info(f"Cloning {url} into {repo_path}...")
                with su.TimeUtils.time_limit(self.CLONE_TIMEOUT):
                    git_process = subprocess.Popen(
                        ["git", "clone", url, str(repo_path)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
                    )
                    stdout, stderr = git_process.communicate()

            else:
                logger.info(f"Repository {repo_name} already exists at {repo_path}")

        except Exception as e:
            su.io.rm(repo_path)
            if git_process and git_process.poll() is None:
                git_process.kill()
            if repo_path.exists():
                su.io.rm(repo_path)
            raise

        return repo_path

    def delete_repository(self, repo_path: su.arg.RPath):
        """Remove a repository directory."""
        try:
            shutil.rmtree(repo_path)
            print(f"Successfully deleted repository at {repo_path}")
        except Exception as e:
            print(f"Failed to delete repository at {repo_path}: {e}")

    def read_urls(self, file_path: su.arg.RPath):
        """Read URLs from a file and return a list of URLs."""
        with open(file_path, "r") as file:
            return [line.strip() for line in file if line.strip()]

    def create_empty_json_file(self, path: su.arg.RPath, file_name: str):
        file_path = path / f"{file_name}.json"

        if file_path.exists():
            print(f"File already exists: {file_path}")
            return None

        # Initialize data as an empty dictionary
        data = {}

        su.io.mkdir(path)

        # Writing empty JSON data to a file
        su.io.dump(file_path, data, su.io.fmts.json)

        return file_path

    def run_refactoring_miner(self, option: str, args: List[str]):
        command = ["RefactoringMiner"]
        if option == "-h":
            command.append("-h")
        elif option == "-a" and len(args) == 3:
            command.extend(["-a", args[0], args[1], "-json", args[2]])
        elif option == "-bc" and len(args) == 4:
            command.extend(["-bc", args[0], args[1], args[2], "-json", args[3]])
        elif option == "-bt" and len(args) == 4:
            command.extend(["-bt", args[0], args[1], args[2], "-json", args[3]])
        elif option == "-c" and len(args) == 3:
            command.extend(["-c", args[0], args[1], "-json", args[2]])
        else:
            return "Invalid command or arguments"

        try:
            result = subprocess.run(command, capture_output=True, text=True, check=True)
            return result.stdout
        except subprocess.CalledProcessError as e:
            return f"An error occurred: {e.stderr}"

    def count_refactorings(self, json_data: Dict[str, Any]):
        # Initialize global refactoring counts
        refactoring_types = {
            "replace loop with pipeline": 0,
            "replace anonymous with lambda": 0,
            "merge catch": 0,
            "replace anonymous with class": 0,
            "replace generic with diamond": 0,
            "try with resources": 0,
        }

        # Flag to check if any refactoring was counted
        any_refactoring_counted = False

        # Process each commit in the JSON data
        for commit in json_data["commits"]:
            for refactoring in commit["refactorings"]:
                ref_type = refactoring["type"].lower()
                if ref_type in refactoring_types:
                    refactoring_types[ref_type] += 1
                    any_refactoring_counted = True  # Set flag if any refactoring is counted

        # Return a boolean signal if any refactorings were counted
        return any_refactoring_counted

    def get_default_branch(self, repo_path: su.arg.RPath):
        # Ensure repo_path is a string, as subprocess works with string paths
        repo_path = str(repo_path)

        # Get the default branch using the command-line Git
        # This command gets the symbolic-ref for HEAD, which points to the default branch
        result = subprocess.run(
            ["git", "remote", "show", "origin"], cwd=repo_path, text=True, capture_output=True, check=True
        )

        # Process the output to find the default branch
        for line in result.stdout.splitlines():
            if "HEAD branch" in line:
                return line.split(":")[1].strip()

    def save_latest_commit(self, json_file_path: su.arg.RPath, latest_commit: str, repo_name: str):
        data = su.io.load(json_file_path, su.io.fmts.json)
        data["latest_commit"] = latest_commit
        su.io.dump(json_file_path, data, su.io.fmts.json)
        logger.info(f"{repo_name}: Latest commit saved successfully.")

    def get_latest_commit(self, repo_path: su.arg.RPath):
        # Change directory to the repository path and get the latest commit hash
        # This command returns the hash of the latest commit
        commit_hash = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_path, text=True
        ).strip()  # Removes any trailing newlines or spaces
        return commit_hash

    def collect_refactoring_worker(self, url: str, file_to_blame: str = "pom.xml") -> bool:
        try:
            parts = url.replace(".git", "").split("/")
            organization = parts[-2]  # Organization is the second last element
            project_name = parts[-1]

            repo_name = f"{organization}_{project_name}"
            repo_path = self.download_dir / repo_name

            json_file_path = self.create_empty_json_file(self.result_refactoring_dir, repo_name)
            if json_file_path is None:
                logger.info(f"Skipping {repo_name} due to existing JSON file.")
                return False

            any_refactoring_counted = False

            self.clone_repository(url, self.download_dir, repo_name)

            full_file_path = repo_path / file_to_blame

            if not full_file_path.exists():
                logger.info(f"The file '{file_to_blame}' does not exist in the repository at '{repo_path}'.")
                return False

            default_branch = self.get_default_branch(repo_path)

            output = self.run_refactoring_miner("-a", [repo_path, default_branch, json_file_path])

            json_data = su.io.load(json_file_path, su.io.fmts.json)

            any_refactoring_counted = self.count_refactorings(json_data)

            if not any_refactoring_counted:
                return False  # Return False if no refactorings were counted

            latest_commit = self.get_latest_commit(repo_path)

            self.save_latest_commit(json_file_path, latest_commit, repo_name)

            return True  # Return True if everything succeeded

        except Exception as e:
            logger.warning(f"An error occurred while processing repository:{url}")
            logger.warning(f"error: {e}")
            return False  # Return False if an exception occurs

        finally:
            if "repo_path" in locals():
                su.io.rm(repo_path)
                logger.info(f"Removed cloned repository at {repo_path}")
            if "json_file_path" in locals() and json_file_path and not any_refactoring_counted:
                su.io.rm(json_file_path)

    def collect_refactoring(
        self,
        max_to_process: Optional[int] = None,
        num_processes: int = 8,
        input_filename: str = "combined_no_fork.json",
    ):
        """
        main function to run
        """
        processed_count = 0
        success_count = 0

        input_filepath = self.raw_repos_dir / input_filename

        repo_urls = su.io.load(input_filepath, su.io.fmts.json)

        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            futures = []

            # Submit tasks to the executor for each repository URL
            for url in repo_urls:
                if max_to_process is not None and processed_count >= max_to_process:
                    break
                future = executor.submit(self.collect_refactoring_worker, url)
                futures.append(future)
                processed_count += 1

            # Collect results from futures
            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result:
                        success_count += 1
                except Exception as exc:
                    print(f"An error occurred: {exc}")

            logger.warning(f"Futures length: {len(futures)}")
            logger.warning("Successfully processed: %d", success_count)
            logger.warning("Total processed: %d", processed_count)


if __name__ == "__main__":
    CLI(RefactoringCollector, as_positional=False)
