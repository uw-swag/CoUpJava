import os
import time
from pathlib import Path
from typing import Dict, Optional, Set

import requests
import seutil as su
from jsonargparse import CLI


class RepoCollector:
    def __init__(
        self,
        token: Optional[str] = None,
        work_dir: su.arg.RPath = Path(__file__).parent.parent.parent / "_work",
        result_repo_name: str = "repos",
    ):
        self.token = token
        self.work_dir = work_dir
        self.result_repo_name = result_repo_name
        self.result_dir = self.work_dir / self.result_repo_name

    def check_rate_limit(self, headers: Dict[str, str]):
        """
        rate limit for search operation
        """
        rate_limit_url = "https://api.github.com/rate_limit"
        response = requests.get(rate_limit_url, headers=headers)
        rate_limits = response.json()
        remaining = rate_limits["rate"]["remaining"]
        reset_time = rate_limits["rate"]["reset"]

        code_search_limit = rate_limits["resources"]["code_search"]
        remaining = code_search_limit["remaining"]
        reset_epoch = int(code_search_limit["reset"])

        reset_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(reset_epoch))

        print(f"Code search rate limit: {code_search_limit['limit']}")
        print(f"Remaining: {remaining}")
        print(f"Resets at: {reset_time}")

        return remaining, reset_epoch

    def search_github(
        self, token: str, search_query: str, file_path: su.arg.RPath, tracked_urls: Set[str], max_results: int = 10000
    ):
        """
        searching the github with rate limit check and wait
        """
        headers = {"Authorization": f"token {token}"}
        base_url = "https://api.github.com/search/code"
        params = {
            "q": search_query,
            "per_page": 100,  # Maximum allowed per page
        }

        with open(file_path, "w") as file:
            total_count = 0
            repos_returned_count = 0
            while total_count < max_results:
                remaining, reset_epoch = self.check_rate_limit(headers)

                if remaining == 0:
                    wait_time = reset_epoch - time.time()
                    print(f"Rate limit reached. Waiting for {wait_time} seconds.")
                    time.sleep(wait_time + 1)  # Wait until the rate limit resets

                print(f"Fetching data from: {base_url}")
                response = requests.get(base_url, headers=headers, params=params)

                if response.status_code == 403:  # Forbidden, likely rate limit hit
                    remaining, reset_epoch = self.check_rate_limit(headers)
                    if remaining == 0:
                        wait_time = reset_epoch - time.time()
                        print(f"Rate limit reached. Waiting for {wait_time} seconds.")
                        time.sleep(wait_time + 1)  # Wait until the rate limit resets
                    continue  # Retry the request after waiting

                if response.status_code != 200:
                    print(f"Failed to fetch data: {response.status_code} - {response.json().get('message')}")
                    time.sleep(10)  # Wait a bit before retrying
                    continue  # Retry the request

                items = response.json().get("items", [])
                repos_returned_count += len(items)
                if not items:
                    print("No more items found.")
                    break

                for item in items:
                    if total_count >= max_results:
                        break
                    # Attempt to extract the clone URL from the repository's HTML URL
                    if item["name"] == "pom.xml":
                        if "repository" in item and "html_url" in item["repository"]:
                            html_url = item["repository"]["html_url"]
                            clone_url = html_url + ".git" if not html_url.endswith(".git") else html_url
                            if clone_url not in tracked_urls:
                                tracked_urls.add(clone_url)
                                file.write(clone_url + "\n")
                                total_count += 1

                # Prepare the next page
                if "next" in response.links:
                    base_url = response.links["next"]["url"]
                else:
                    print("No more pages found.")
                    print(response.links)
                    break

        print(f"Saved {total_count} repository URLs to '{file_path}'.")
        print(f"Total repositories returned by responses: {repos_returned_count}")

    def collect_repo(self, max_results=10000):
        """
        main function
        """
        if self.token is None:
            raise ValueError("A GitHub authorization token is required")

        search_queries = []
        target_dir = self.result_dir

        for i in range(7, 24):
            query_keyword_1 = f"<maven.compiler.source>{i}</maven.compiler.source> filename:pom extension:xml path:/"
            save_filepath_1 = os.path.join(target_dir, f"repos_to_clone_java_{i}.txt")
            search_queries.append((query_keyword_1, save_filepath_1))
        for i in range(7, 11):
            query_keyword_2 = f"<maven.compiler.source>1.{i}</maven.compiler.source> filename:pom extension:xml path:/"
            save_filepath_2 = os.path.join(target_dir, f"repos_to_clone_java_1.{i}.txt")
            search_queries.append((query_keyword_2, save_filepath_2))

        # Print the list of search queries
        print("list of search queries:")
        for query in search_queries:
            print(query)

        tracked_urls = set()
        for search_query, save_filepath in search_queries:
            if os.path.exists(save_filepath):
                with open(save_filepath, "r") as file:
                    for line in file:
                        url = line.strip().replace(".git", "")
                        tracked_urls.add(url)
            else:
                self.search_github(self.token, search_query, save_filepath, tracked_urls, max_results)

    def count_repos(self):
        all_files = []
        for file_path in self.result_dir.iterdir():
            if file_path.is_file():
                all_files.append(file_path)
        count = 0
        for file_path in all_files:
            file_list = su.io.load(file_path, su.io.fmts.txtList)
            length = len(file_list)
            bash_res = su.bash.run(f"wc -l {file_path}")
            print(bash_res.stdout)
            count += length

        print(count)

    def check_rate_limit_core(self):
        """
        rate limit for core operation
        """
        url = "https://api.github.com/rate_limit"
        headers = {
            "Authorization": f"token {self.token}"  # Optional, for higher rate limits
        }

        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            # Extract rate limit headers
            rate_limit_remaining = int(response.headers.get("X-RateLimit-Remaining", 0))
            reset_timestamp = int(response.headers.get("X-RateLimit-Reset", 0))

            if rate_limit_remaining == 0:
                # Calculate the time to wait in seconds if the limit is reached
                current_time = int(time.time())
                wait_time = reset_timestamp - current_time

                if wait_time > 0:
                    print(f"Rate limit reached. Waiting for {wait_time} seconds...")
                    time.sleep(wait_time)  # Pause the script until the reset time
                    print("Resuming operations after the wait.")
                else:
                    print("Rate limit reset time has already passed. Continuing immediately.")
            else:
                print(f"Requests remaining: {rate_limit_remaining}. No need to wait.")
        else:
            print(f"Failed to retrieve rate limit information. Status code: {response.status_code}")

    def is_fork(self, organization: str, project_name: str):
        headers = {"Authorization": f"token {self.token}"}

        url = f"https://api.github.com/repos/{organization}/{project_name}"
        while True:
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                data = response.json()
                return data.get("fork")
            elif response.status_code == 403:
                self.check_rate_limit_core()
            elif response.status_code == 404:
                return True
            else:
                print(f"Failed to retrieve {organization}/{project_name}. Status code: {response.status_code}")
                time.sleep(10)

    def combine_raw_repo(self):
        raw_repo_filtered_dir = self.work_dir / "_raw-repo"
        all_files = []
        for file_path in raw_repo_filtered_dir.iterdir():
            if file_path.is_file():
                all_files.append(file_path)
        for file_path in all_files:
            print(file_path)
        combined_list = []
        for file_path in all_files:
            file_list = su.io.load(file_path, su.io.fmts.txtList)
            combined_list.extend(file_list)
        output_path = raw_repo_filtered_dir / "combined.json"
        su.io.dump(output_path, combined_list, su.io.fmts.json)

    def remove_fork_combined_file(self, file_name: str = "combined.json"):
        file_path = self.result_dir / file_name
        print(file_path)
        file_list = su.io.load(file_path, su.io.fmts.json)
        length = len(file_list)
        print(length)
        result_list = []
        for url in file_list:
            parts = url.replace(".git", "").split("/")
            organization = parts[-2]  # Organization is the second last element
            project_name = parts[-1]
            ret = self.is_fork(organization, project_name)
            if ret is False:
                result_list.append(url)
            else:
                print(f"{url} is fork repo")
        print(f"{len(result_list)} out of {length}")
        output_name = "combined_no_fork.json"
        output_path = self.work_dir / "_raw-repo" / output_name
        su.io.dump(output_path, result_list, su.io.fmts.json)


if __name__ == "__main__":
    CLI(RepoCollector, as_positional=False)
