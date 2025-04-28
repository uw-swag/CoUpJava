import difflib
import json
import logging
import os
import subprocess
from pathlib import Path
from typing import Dict, List

import seutil as su
from jsonargparse import CLI

logger = su.log.get_logger(__name__)


class DatasetBuilder:
    def __init__(
        self,
        work_dir: su.arg.RPath = Path(__file__).parent.parent.parent / "_work",
        download_repo_name: str = "_downloads",
        stats_repo_name: str = "stats",
        test_repo_name: str = "test",
        output_dir_name: str = "data",
        input_dir_name: str = "raw-data",
        clone_timeout: int = 300,
        logger_name: str = "build_dataset.log",
    ):
        self.work_dir = work_dir
        self.download_repo_name = download_repo_name
        self.download_dir = self.work_dir / self.download_repo_name
        self.stats_repo_name = stats_repo_name
        self.stats_dir = self.work_dir / self.stats_repo_name
        self.test_repo_name = test_repo_name
        self.test_dir = self.work_dir / self.stats_repo_name
        self.output_dir_name = output_dir_name
        self.output_dir = self.work_dir / self.output_dir_name
        self.input_dir_name = input_dir_name
        self.input_dir = self.work_dir / self.input_dir_name
        self.CLONE_TIMEOUT = clone_timeout

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
        directories = [self.download_dir, self.stats_dir, self.test_dir, self.input_dir, self.output_dir]
        for dir_path in directories:
            if not dir_path.exists():
                dir_path.mkdir(parents=True, exist_ok=True)
                logger.info(f"Created directory: {dir_path}")
            else:
                logger.info(f"Directory already exists: {dir_path}")

    def extract_refactorings(self, commits: List[Dict], refactoring_types: List[str]):
        """
        Function to extract relevant information
        """
        filter_criteria = {
            "replace loop with pipeline": {"side": "leftSideLocations", "required_element_type": "FOR_STATEMENT"}
            # Add more refactoring types with their specific criteria here
        }

        # Initialize the discarded counts for each type with filtering criteria
        discarded_counts = {key: 0 for key in filter_criteria}

        results = []
        for commit in commits:
            repo_name = commit["repository"]
            commit_hash = commit["sha1"]
            repo_url = commit["url"]
            for ref_index, refactoring in enumerate(commit["refactorings"]):
                ref_type = refactoring["type"].lower()
                if ref_type in map(str.lower, refactoring_types):
                    instance_details = {
                        "repo_name": repo_name,
                        "hash": commit_hash,
                        "repo_url": repo_url,
                        "ref_type": ref_type,
                        "description": refactoring.get("description", "No description"),
                        "leftSideLocations": [],
                        "rightSideLocations": [],
                    }

                    # Populate locations
                    for side in ["leftSideLocations", "rightSideLocations"]:
                        for loc in refactoring[side]:
                            location = {
                                "filePath": loc["filePath"],
                                "startLine": loc["startLine"],
                                "endLine": loc["endLine"],
                                "startColumn": loc["startColumn"],
                                "endColumn": loc["endColumn"],
                                "codeElementType": loc["codeElementType"],
                                "description": loc["description"],
                                "codeElement": loc.get("codeElement", "No code element"),
                            }
                            instance_details[side].append(location)

                    # Apply filters based on predefined criteria
                    if ref_type in filter_criteria:
                        criteria = filter_criteria[ref_type]
                        required_type_found = any(
                            loc["codeElementType"] == criteria["required_element_type"]
                            for loc in instance_details[criteria["side"]]
                        )
                        if not required_type_found:
                            discarded_counts[ref_type] += 1
                            continue  # Skip adding this refactoring instance to results

                    # Append the instance details to the results list
                    results.append(instance_details)
        logger.info(f"discarded_counts: {discarded_counts}")
        return results

    def merge_ranges(self, ranges: List[Dict]):
        """
        merge non METHOD_DECLARATION lines
        """
        if not ranges:
            return []

        # Sort ranges by start line for sequential processing
        sorted_ranges = sorted(ranges, key=lambda x: x["startLine"])

        # Initialize the list to store merged ranges
        merged = []

        # Iterate over sorted ranges and merge where appropriate
        for current in sorted_ranges:
            # Handle METHOD_DECLARATION separately
            if current["codeElementType"] == "METHOD_DECLARATION":
                merged.append(current)
            else:
                if not merged:
                    merged.append(current)
                else:
                    last = merged[-1]
                    # Check if the last merged range is also not a METHOD_DECLARATION
                    if last["codeElementType"] != "METHOD_DECLARATION":
                        # Check if the current range overlaps with or is adjacent to the last merged range
                        if current["startLine"] <= last["endLine"] + 1:
                            # Merge the current range into the last range in 'merged'
                            merged[-1]["endLine"] = max(last["endLine"], current["endLine"])
                            # Optionally merge other fields as needed, e.g., endColumn
                            merged[-1]["endColumn"] = max(last["endColumn"], current["endColumn"])
                        else:
                            merged.append(current)
                    else:
                        merged.append(current)

        return merged

    def process_refactorings_data(self, refactoring_data: List[Dict]):
        """
        call merge_ranges on all refactoring data
        """
        processed_data = []
        for instance_details in refactoring_data:
            for side in ["leftSideLocations", "rightSideLocations"]:
                instance_details[side] = self.merge_ranges(instance_details[side])
            processed_data.append(instance_details)
        return processed_data

    # Function to extract lines between start_line and end_line
    def extract_lines(self, start_line: int, end_line: int, lines_of_code: List[str]):
        # Adjust for 0-based index (Python lists are 0-based but line numbers are 1-based)
        return lines_of_code[start_line - 1 : end_line]

    def extract_src_code(self, commit_hash: str, repo_path: su.arg.RPath, file_path: su.arg.RPath, before_flag: bool):
        """
        extract src code of a file
        """
        # Determine which commit to use based on before_flag
        commit_to_use = commit_hash + "^" if before_flag else commit_hash

        # Construct the git show command
        cmd = ["git", "show", f"{commit_to_use}:{file_path}"]
        # Run the git command
        result = subprocess.run(cmd, cwd=repo_path, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        # Split the result by newlines to get the source code lines
        if result.returncode != 0:
            logger.error(f"Git command failed: {result.stderr}")
            return None
        return result.stdout.split("\n")

    def extract_and_merge_lines(
        self,
        commit_hash: str,
        repo_path: su.arg.RPath,
        file_path: su.arg.RPath,
        location: Dict,
        side: str,
        ref_type: str,
    ):
        """
        extraced codes specified by locations
        """
        # Determine the before_flag based on the side of the refactoring
        before_flag = True if side == "leftSideLocations" else False

        # Extract source code based on the specified side
        lines_of_code = self.extract_src_code(commit_hash, repo_path, file_path, before_flag)

        if not lines_of_code:
            logger.info(f"No lines of code found for {file_path}, skipping.")
            return None

        if location["startLine"] - 1 < len(lines_of_code) and location["endLine"] <= len(lines_of_code):
            combined_lines = lines_of_code[location["startLine"] - 1 : location["endLine"]]
        else:
            combined_lines = []

        # Create a JSON object to store both the text version and structured line data
        data_to_save = {
            # "extracted_text": "\n".join(combined_lines),
            "extracted_lines": combined_lines,
            "startLine": location["startLine"],
            "endLine": location["endLine"],
        }

        return data_to_save

    def extract_lines_from_code(self, lines_of_code: List[str], start: int, end: int):
        # Ensure that 'lines_of_code' is a list of strings and 'start' and 'end' are integers
        if not lines_of_code or start > end:
            return []
        return lines_of_code[start - 1 : end]

    def find_method_location(self, locations: List[Dict]):
        for index, location in enumerate(locations):
            if location["codeElementType"] == "METHOD_DECLARATION":
                return location, index
        return None, None

    def process_right_side_lines(
        self, commit_hash: str, repo_path: su.arg.RPath, file_path: su.arg.RPath, instance_details: Dict
    ):
        # Extract source code based on the specified side
        left_file_path = instance_details["leftSideLocations"][0]["filePath"]
        left_file_code = self.extract_src_code(commit_hash, repo_path, left_file_path, True)
        right_file_code = self.extract_src_code(commit_hash, repo_path, file_path, False)

        right_extracted_data = instance_details["extracted_data"]["rightSideLocations"]

        right_method = None
        for data in right_extracted_data:
            if data["codeElementType"] == "METHOD_DECLARATION":
                right_method = data
            else:
                right_refactoring = data

        if right_method is None:
            logger.info("right_method is None: no method declartion found")
            return None, None

        left_extracted_data = instance_details["extracted_data"]["leftSideLocations"]

        left_method = None
        for data in left_extracted_data:
            if data["codeElementType"] == "METHOD_DECLARATION":
                left_method = data
            else:
                left_refactoring = data

        if left_method is None:
            logger.info("left_method is None: no method declartion found")
            return None, None

        part1 = self.extract_lines_from_code(
            right_file_code, right_method["startLine"], right_refactoring["startLine"] - 1
        )
        part2 = self.extract_lines_from_code(left_file_code, left_refactoring["startLine"], left_refactoring["endLine"])
        part3 = self.extract_lines_from_code(right_file_code, right_refactoring["endLine"] + 1, right_method["endLine"])

        old_data = part1 + part2 + part3

        new_data = self.extract_lines_from_code(right_file_code, right_method["startLine"], right_method["endLine"])
        return new_data, old_data

    def process_and_save_results(self, refactoring_data: List[Dict], repo_path: su.arg.RPath, output_dir: su.arg.RPath):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        results = []

        for instance_details in refactoring_data:
            commit_hash = instance_details["hash"]
            ref_type = instance_details["ref_type"]

            instance_details["extracted_data"] = {"leftSideLocations": [], "rightSideLocations": []}
            instance_details["processed_data"] = {"old_data": [], "new_data": []}

            instance_details["extracted_data_with_context"] = {"leftSideLocations": [], "rightSideLocations": []}

            # Extract and store data for both sides
            for side in ["leftSideLocations", "rightSideLocations"]:
                locations = instance_details[side]
                for location in locations:
                    src_file_path = location["filePath"]
                    codeElementType = location["codeElementType"]
                    extracted_data = self.extract_and_merge_lines(
                        commit_hash, repo_path, src_file_path, location, side, ref_type
                    )
                    if extracted_data:
                        # instance_details['extracted_data'][side].append(extracted_data)
                        data_with_context = {
                            "codeElementType": codeElementType,
                            "extracted_lines": extracted_data["extracted_lines"],
                            "startLine": extracted_data["startLine"],
                            "endLine": extracted_data["endLine"],
                        }
                        instance_details["extracted_data"][side].append(data_with_context)

                # Special processing for right side
                if side == "rightSideLocations":
                    new_data, old_data = self.process_right_side_lines(
                        commit_hash, repo_path, src_file_path, instance_details
                    )

                    if new_data is None or old_data is None:
                        continue
                    instance_details["processed_data"]["new_data"] = new_data
                    instance_details["processed_data"]["old_data"] = old_data
            results.append(instance_details)

        return results

    def save_dataset_as_jsonl(self, dataset: List[Dict], target_dir: su.arg.RPath, append: bool = False):
        su.io.dump(target_dir, dataset, su.io.fmts.jsonList, append=append)

    def clone_repository(self, url: str, target_dir: su.arg.RPath, repo_name: str):
        """Clone a single repository into the specified directory using command-line Git."""
        repo_path = target_dir / repo_name
        try:
            if not repo_path.exists():
                logger.info(f"Cloning {url} into {repo_path}...")
                with su.TimeUtils.time_limit(self.CLONE_TIMEOUT):
                    clone_cmd = f"git clone {url} {repo_path}"
                    result = su.bash.run(clone_cmd)
                    logger.info(f"{result.stdout}")
            else:
                logger.info(f"Repository {repo_name} already exists at {repo_path}")

        except Exception as e:
            su.io.rm(repo_path)
            logger.error(f"Failed to clone repository {url}: {e}")
            raise Exception(f"Cloning failed for {url}")

    def is_directory_empty(self, directory_path: su.arg.RPath):
        # Check if the directory is empty
        return not any(directory_path.iterdir())

    def build_dataset_single_process(self, output_filename: str = "refactoringminer_dataset_new.jsonl"):
        """
        main function to build dataset
        """
        # Sample JSON data (you would replace 'data' with your actual JSON string)
        output_path = self.output_dir / output_filename
        # Open the file and load the data
        refactoring_types = [
            "replace loop with pipeline",
            "replace anonymous with lambda",
            "merge catch",
            "replace anonymous with class",
            "replace generic with diamond",
            "try with resources",
        ]
        files = list(self.input_dir.iterdir())
        length = len(files)
        logger.info(f"num of files: {length}")
        sum = 0
        for idx, file_path in enumerate(files):
            try:
                logger.info(f"start processing: {file_path}, idx:{idx}")
                data = su.io.load(file_path, su.io.fmts.json)

                url = data["commits"][0]["repository"]
                parts = url.replace(".git", "").split("/")
                organization = parts[-2]  # Organization is the second last element
                project_name = parts[-1]

                repo_name = f"{organization}_{project_name}"
                repo_path = self.download_dir / repo_name
                self.clone_repository(url, self.download_dir, repo_name)

                refactoring_data = self.extract_refactorings(data["commits"], refactoring_types)
                sum += len(refactoring_data)
                processed_data = self.process_refactorings_data(refactoring_data)

                if (not repo_path.exists()) or (self.is_directory_empty(repo_path)):
                    su.io.rm(repo_path)
                    continue

                results = self.process_and_save_results(processed_data, repo_path, self.output_dir)

                self.save_dataset_as_jsonl(results, output_path, append=True)

            except Exception as e:
                logger.warning(f"An error occurred while processing repository:{url}", exc_info=True)
                logger.warning(f"error: {e}")

            finally:
                if "repo_path" in locals():
                    su.io.rm(repo_path)
                    logger.info(f"Removed cloned repository at {repo_path}")

        logger.info(f"total count: {sum}")

    def collect_stat_on_repos_by_type(self, input_filename: str = "refactoringminer_dataset_combined.jsonl"):
        """
        collect git diff vs refactoringminer diff stats on dataset
        """
        # Initialize a dictionary to store counts per refactoring type
        ref_type_stats = {}

        input_path = self.output_dir / input_filename
        less_count = 0
        equal_count = 0
        dataset = su.io.load(input_path, su.io.fmts.jsonList)
        print(len(dataset))
        for datapoint in dataset:
            url = datapoint["repo_name"]
            parts = url.replace(".git", "").split("/")
            organization = parts[-2]  # Organization is the second last element
            project_name = parts[-1]
            pretty_json = json.dumps(datapoint, indent=4, sort_keys=True)
            # print(pretty_json)
            repo_name = f"{organization}_{project_name}"
            repo_path = self.download_dir / repo_name

            # self.clone_repository(url, self.download_dir, repo_name)
            type_stats = self.collect_stat_by_type(datapoint, repo_path)  # Collect stats per type

            # Aggregate statistics for each type across all files
            for ref_type, stats in type_stats.items():
                if ref_type not in ref_type_stats:
                    ref_type_stats[ref_type] = {"total": 0, "more": 0, "equal": 0, "less": 0}
                ref_type_stats[ref_type]["total"] += stats["total"]
                ref_type_stats[ref_type]["more"] += stats["more"]
                ref_type_stats[ref_type]["equal"] += stats["equal"]
                ref_type_stats[ref_type]["less"] += stats["less"]
                less_count += stats["less"]
                equal_count += stats["equal"]

        print(less_count + equal_count)
        print(ref_type_stats)
        return ref_type_stats

    def build_dataset_filtered(
        self,
        input_filename: str = "refactoringminer_dataset_new.jsonl",
        output_filename: str = "refactoringminer_dataset_new_filtered.jsonl",
    ):
        """
        filter the dataset such that datapoint whose git diff has equal or less lines than refactoringminer diff will be kept
        """
        input_path = self.output_dir / input_filename
        output_path = self.output_dir / output_filename
        dataset = su.io.load(input_path, su.io.fmts.jsonList)
        filtered_dataset = []
        for datapoint in dataset:
            url = datapoint["repo_name"]
            parts = url.replace(".git", "").split("/")
            organization = parts[-2]  # Organization is the second last element
            project_name = parts[-1]
            repo_name = f"{organization}_{project_name}"

            self.filter_data(datapoint, filtered_dataset)  # Collect stats per type

        self.save_dataset_as_jsonl(filtered_dataset, output_path)

        print(f"result len: {len(filtered_dataset)}")

    def filter_data(self, datapoint: Dict, filtered_dataset: List):
        type_stats = {}
        ref_type = datapoint["ref_type"]

        left_method = None
        right_method = None
        refactoring_deleted = 0
        refactoring_added = 0

        # Initialize stats dictionary for the ref_type if not already present
        if ref_type not in type_stats:
            type_stats[ref_type] = {"more": 0, "less": 0, "equal": 0, "total": 0}

        for side in ["leftSideLocations", "rightSideLocations"]:
            extracted_data = datapoint["extracted_data"][side]
            for data in extracted_data:
                # Use dict.get() to safely handle missing 'codeElementType'
                code_element_type = data.get("codeElementType")
                if code_element_type is None:
                    # Print the datapoint and skip if 'codeElementType' is missing
                    print(f"Missing 'codeElementType' in: {ref_type}")
                    continue

                if code_element_type == "METHOD_DECLARATION":
                    if side == "leftSideLocations":
                        left_method = data["extracted_lines"]
                    elif side == "rightSideLocations":
                        right_method = data["extracted_lines"]
                else:
                    line_count = data["endLine"] - data["startLine"] + 1
                    if side == "leftSideLocations":
                        refactoring_deleted += line_count
                    elif side == "rightSideLocations":
                        refactoring_added += line_count

        if left_method and right_method:
            added, deleted = self.compare_code_lists_and_count_changes(left_method, right_method)

            if added == refactoring_added and deleted == refactoring_deleted:
                filtered_dataset.append(datapoint)
            elif added < refactoring_added and deleted < refactoring_deleted:
                filtered_dataset.append(datapoint)

    def compare_code_lists_and_count_changes(self, list1: List, list2: List):
        """Compare two lists of lines of code, ignoring differences in indentation, and separately count additions and deletions."""
        # Normalize both lists
        normalized_list1 = self.normalize_lines(list1)
        normalized_list2 = self.normalize_lines(list2)

        # Generate the unified diff
        diff = list(difflib.unified_diff(normalized_list1, normalized_list2, lineterm="", n=0))

        # Initialize counters for additions and deletions
        additions = 0
        deletions = 0

        # Parse the diff to count actual changes, specifically filtering out non-change lines
        for line in diff:
            # print(line)  # Print each line of the diff
            if line.startswith("+") and not line.startswith("+++ "):
                additions += 1
            elif line.startswith("-") and not line.startswith("--- "):
                deletions += 1

        return additions, deletions

    def get_commit_timestamp(self, repo_path: su.arg.RPath, commit_hash: str):
        """
        Get the commit timestamp in hexadecimal format for a specific commit hash
        in a given Git repository path.

        :param repo_path: Path to the Git repository.
        :param commit_hash: The specific commit hash to retrieve the timestamp.
        :return: Commit timestamp in hexadecimal format.
        """
        # Construct the git command to get the Unix timestamp
        git_command = ["git", "-C", repo_path, "show", "-s", "--format=%ct", commit_hash]

        try:
            result = subprocess.run(git_command, capture_output=True, text=True, check=True)
            unix_timestamp = result.stdout.strip()

            return unix_timestamp

        except subprocess.CalledProcessError as e:
            print(f"Error occurred while running git command: {e}")
            return None

    def normalize_lines(self, lines: List):
        """Normalize the indentation of lines by stripping leading whitespace."""
        return [line.lstrip() for line in lines]

    def combine_jsonl_from_repo(
        self, repo_path: su.arg.RPath = None, output_file_name: str = "refactoringminer_dataset_combined.jsonl"
    ):
        """
        Combines all JSONL files in a specified directory into a single JSONL file.

        Parameters:
            repo_path (str): Path to the directory containing JSONL files.
            output_file (str): Path to the output combined JSONL file.
        """
        # List all .jsonl files in the specified directory
        if repo_path is None:
            repo_path = self.output_dir
        input_files = [f for f in repo_path.iterdir() if f.is_file() and f.suffix == ".jsonl"]
        if not input_files:
            logger.info("No JSONL files found in the specified directory.")
            return
        print(input_files)
        combined_data = []
        for input_file in input_files:
            try:
                data = su.io.load(input_file, su.io.fmts.jsonList)
                logger.info(f"len data {len(data)}")
                combined_data.extend(data)  # Assuming `data` is a list of JSON objects
            except Exception as e:
                logger.error(f"Failed to load {input_file}: {e}")
                continue

        output_path = repo_path / output_file_name
        logger.info(f"combined len:{len(combined_data)}")
        self.save_dataset_as_jsonl(combined_data, output_path)

    def remove_invalid_timestamp(
        self,
        input_filename: str = "refactoringminer_dataset_new_filtered_timestamp.jsonl",
        output_filename: str = "refactoringminer_dataset_new_filtered_timestamp.jsonl",
    ):
        """
        fill any missing timestamp that caused by errors when using build_dataset_with_timestamp
        """
        input_path = self.output_dir / input_filename
        output_path = self.output_dir / output_filename
        dataset = su.io.load(input_path, su.io.fmts.jsonList)
        final_dataset = []

        for datapoint in dataset:
            if datapoint["commit_timestamp"] is None:
                continue

            final_dataset.append(datapoint)

        self.save_dataset_as_jsonl(final_dataset, output_path)

    def remove_duplicate(
        self,
        input_filename: str = "refactoringminer_dataset_new_filtered_timestamp.jsonl",
        output_filename: str = "refactoringminer_dataset_new_filtered_timestamp_nodup.jsonl",
    ):
        """
        remove duplication in dataset
        """
        new_code_dict = {}
        input_path = self.output_dir / input_filename
        output_path = self.output_dir / output_filename
        dataset = su.io.load(input_path, su.io.fmts.jsonList)
        unduplicated_dataset = []
        duplicate_count = 0

        new_dataset = []
        for data_point in dataset:
            repo_name = data_point["repo_name"]
            parts = repo_name.split("_")
            organization = parts[0]
            project_name = parts[1]
            url = f"https://github.com/{organization}/{project_name}.git"
            new_dataset.append(data_point)

        for data_point in new_dataset:
            # Convert the list of code lines into single strings
            new_code_str = "\n".join(data_point["new_code"])
            old_code_str = "\n".join(data_point["old_code"])
            # Use a tuple of (new_code_str, old_code_str) as the key to detect duplicates
            code_pair = (new_code_str, old_code_str)

            if code_pair not in new_code_dict:
                # First occurrence, store the data point
                new_code_dict[code_pair] = data_point
                unduplicated_dataset.append(data_point)
            else:
                # Duplicate found
                duplicate_count += 1

        print(f"Number of duplicates removed: {duplicate_count}")
        # Save the unduplicated dataset

        self.save_dataset_as_jsonl(unduplicated_dataset, output_path, append=True)

    def count_unique_repos(self, input_filename: str = "refactoringminer_dataset_combined.jsonl"):
        input_path = self.output_dir / input_filename
        dataset = su.io.load(input_path, su.io.fmts.jsonList)
        repo_set = set()
        for data_point in dataset:
            repo_set.add(data_point["repo_name"])  # Corrected here

        print(len(repo_set))

    def build_dataset_with_timestamp(
        self,
        input_filename: str = "refactoringminer_dataset_new_filtered.jsonl",
        output_filename: str = "refactoringminer_dataset_new_filtered_timestamp.jsonl",
    ):
        """
        add hash timestamp to dataset
        """
        input_path = self.output_dir / input_filename
        output_path = self.output_dir / output_filename
        dataset = su.io.load(input_path, su.io.fmts.jsonList)
        final_dataset = []
        prev_url = None
        prev_repo_path = None

        for datapoint in dataset:
            old_code = ""
            for data in datapoint["extracted_data"]["leftSideLocations"]:
                if data["codeElementType"] == "METHOD_DECLARATION":
                    old_code = data["extracted_lines"]
                    break
            new_code = ""

            for data in datapoint["extracted_data"]["rightSideLocations"]:
                if data["codeElementType"] == "METHOD_DECLARATION":
                    new_code = data["extracted_lines"]
                    break

            url = datapoint["repo_name"]
            ref_type = datapoint["ref_type"]
            commit_hash = datapoint["hash"]
            description = datapoint["description"]

            parts = url.replace(".git", "").split("/")
            organization = parts[-2]  # Organization is the second last element
            project_name = parts[-1]

            repo_name = f"{organization}_{project_name}"
            repo_path = self.download_dir / repo_name
            if prev_url != url:
                self.clone_repository(url, self.download_dir, repo_name)
                prev_url = url
            timestamp = self.get_commit_timestamp(repo_path, commit_hash)
            new_instance = {
                "repo_name": repo_name,
                "ref_type": ref_type,
                "Description": description,
                "commit_hash": commit_hash,
                "commit_timestamp": timestamp,
                "new_code": new_code,
                "old_code": old_code,
            }
            final_dataset.append(new_instance)
            if prev_repo_path != repo_path:
                if prev_repo_path is not None:
                    su.io.rm(prev_repo_path)
                prev_repo_path = repo_path

        self.save_dataset_as_jsonl(final_dataset, output_path)

    def check_dropped_repos(self, input_filename: str = "refactoringminer_dataset_new_filtered_timestamp_nodup.jsonl"):
        input_path = self.output_dir / input_filename
        dataset = su.io.load(input_path, su.io.fmts.jsonList)
        dropped_repo_path = self.work_dir / "_raw-repo" / "dropped_repos.json"
        dropped_repo_list = su.io.load(dropped_repo_path, su.io.fmts.json)

        for data_point in dataset:
            repo_name = data_point["repo_name"]
            parts = repo_name.split("_")
            organization = parts[0]
            project_name = parts[1]
            url = f"https://github.com/{organization}/{project_name}.git"
            if url in dropped_repo_list:
                print(f"{repo_name} is java 5 or 6")

    def filter_dropped_repos(
        self,
        input_filename: str = "refactoringminer_dataset_new_filtered_timestamp_nodup.jsonl",
        output_filename: str = "refactoringminer_dataset_new_filtered_timestamp_nodup_final.jsonl",
    ):
        input_path = self.output_dir / input_filename
        dataset = su.io.load(input_path, su.io.fmts.jsonList)
        dropped_repo_path = self.work_dir / "_raw-repo" / "dropped_repos.json"
        dropped_repo_list = su.io.load(dropped_repo_path, su.io.fmts.json)
        output_path = self.output_dir / output_filename

        result_dataset = []
        for data_point in dataset:
            repo_name = data_point["repo_name"]
            parts = repo_name.split("_")
            organization = parts[0]
            project_name = parts[1]
            url = f"https://github.com/{organization}/{project_name}.git"
            if url not in dropped_repo_list:
                # print(f"{repo_name} is java 5 or 6")
                result_dataset.append(data_point)

        su.io.dump(output_path, result_dataset, su.io.fmts.jsonList)

    def rebuild_old_dataset(
        self,
        input_filename: str = "refactoringminer_dataset_final_complete_nodup.jsonl",
        output_filename: str = "refactoringminer_dataset_old_filtered_timestamp_nodup_final.jsonl",
    ):
        input_filepath = self.output_dir / input_filename
        output_filepath = self.output_dir / output_filename

        new_repo_list_path = self.work_dir / "_raw-repo" / "combined_final.json"
        new_repo_list = su.io.load(new_repo_list_path, su.io.fmts.json)

        result_list = []

        input_file = su.io.load(input_filepath, su.io.fmts.jsonList)
        for data_point in input_file:
            repo_name = data_point["repo_name"]
            parts = repo_name.split("_")
            organization = parts[0]
            project_name = parts[1]
            url = f"https://github.com/{organization}/{project_name}.git"

            if url in new_repo_list:
                result_list.append(data_point)

        su.io.dump(output_filepath, result_list, su.io.fmts.jsonList)

    def concat_dataset(
        self,
        input_filename1: str = "refactoringminer_dataset_old_filtered_timestamp_nodup_final.jsonl",
        input_filename2: str = "refactoringminer_dataset_new_filtered_timestamp_nodup_final.jsonl",
        output_filename: str = "refactoringminer_dataset_combined_filtered_timestamp_nodup_final.jsonl",
    ):
        input_filepath1 = self.output_dir / input_filename1
        input_filepath2 = self.output_dir / input_filename2
        output_path = self.output_dir / output_filename
        list1 = su.io.load(input_filepath1, su.io.fmts.jsonList)
        list2 = su.io.load(input_filepath2, su.io.fmts.jsonList)

        result = []

        for data_point in list1:
            result.append(data_point)

        for data_point in list2:
            result.append(data_point)

        su.io.dump(output_path, result, su.io.fmts.jsonList)

    def reformat_dataset(
        self,
        input_filename: str = "refactoringminer_dataset_new_filtered_timestamp_nodup.jsonl",
        output_filename: str = "coupjava-fine.jsonl",
    ):
        input_filepath = self.output_dir / input_filename
        output_filepath = self.output_dir / output_filename
        dataset = su.io.load(input_filepath, su.io.fmts.jsonList)
        for data_point in dataset:
            new_code = data_point["new_code"]
            old_code = data_point["old_code"]
            data_point["new_code"] = "\n".join(new_code)
            data_point["old_code"] = "\n".join(old_code)
            data_point["description"] = data_point.pop("Description")
        su.io.dump(output_filepath, dataset, su.io.fmts.jsonList)

    def inspect_reformat(
        self,
        input_filename1: str = "refactoringminer_dataset_combined_filtered_timestamp_nodup_final.jsonl",
        input_filename2: str = "refactoringminer_dataset_combined_filtered_timestamp_nodup_reformatted_final.jsonl",
    ):
        input_filepath1 = self.output_dir / input_filename1
        input_filepath2 = self.output_dir / input_filename2
        dataset1 = su.io.load(input_filepath1, su.io.fmts.jsonList)
        dataset2 = su.io.load(input_filepath2, su.io.fmts.jsonList)
        for data_point in dataset1[:1]:
            new_code = data_point["new_code"]
            old_code = data_point["old_code"]
            print("dataset1 new_code")
            for line in new_code:
                print(line)
            print("dataset1 old_code")

            for line in old_code:
                print(line)

        for data_point in dataset2[:1]:
            new_code = data_point["new_code"]
            old_code = data_point["old_code"]
            print("dataset2 new_code")
            print(new_code)
            print("dataset2 old_code")
            print(old_code)


if __name__ == "__main__":
    CLI(DatasetBuilder, as_positional=False)
