import collections
import dataclasses
import random
import re
import traceback
from pathlib import Path
from typing import List, Optional

import seutil as su
import unidiff
from jsonargparse import CLI
from seutil.project import Project

from codeupgrade.utils import ALL_JAVA_VERSIONS, JavaEnvManager, normalize_java_version

logger = su.log.get_logger(__name__)


@dataclasses.dataclass
class JavaVersionData:
    repo_name: str = None
    old_commit: str = None
    old_timestamp: str = None
    old_version: int = None
    new_commit: str = None
    new_timestamp: str = None
    new_version: int = None
    patch: str = None


@dataclasses.dataclass
class JavaVersionInterval:
    """
    Representing that the specific java version is used during the interval [beg_commit, end_commit]
    (both end inclusive).
    """

    version: int = None
    beg_commit: str = None
    end_commit: str = None


class JavaVersionDatasetCollector:
    def __init__(
        self,
        work_dir: su.arg.RPath = Path(__file__).parent.parent.parent / "_work",
        downloads_subdir: str = "_downloads",
        data_subdir: str = "data",
        repos_subpath: str = "repos/combined_final.json",
    ):
        self.work_dir = work_dir
        self.downloads_dir = self.work_dir / downloads_subdir
        self.data_dir = self.work_dir / data_subdir
        self.repos_path = self.work_dir / repos_subpath

        self.java_env_manager = JavaEnvManager(self.work_dir / "javaenv")
        self.counter = collections.Counter()
        self.counter["no_interval"] = 0
        self.counter["one_interval"] = 0
        self.counter["downgrade_version"] = 0
        self.counter["clone_failed"] = 0
        self.counter["compilation_failed"] = 0
        self.counter["other_failed"] = 0

    def collect_repos(
        self,
        beg: Optional[int] = None,
        end: Optional[int] = None,
        only: Optional[List[str]] = None,
        debug_sample: Optional[int] = None,
        suffix: Optional[str] = None,
    ):
        repos = self._load_repos()
        if beg is not None or end is not None:
            if beg is None:
                beg = 0
            if end is None:
                end = len(repos)
            logger.info(f"Processing repos {beg}:{end}")
            repos = repos[beg:end]
        if only is not None:
            repos = [repo for repo in repos if repo.full_name in only]
        if debug_sample is not None:
            repos = random.sample(repos, min(debug_sample, len(repos)))

        dataset: List[JavaVersionData] = []
        pbar = su.pbar.tqdm(total=len(repos))
        for repo in repos:
            pbar.set_description(f"{repo.full_name} ({len(dataset)})")
            try:
                try:
                    repo.clone(self.downloads_dir)
                except KeyboardInterrupt:
                    raise
                except Exception:
                    logger.error(f"Cannot clone {repo.full_name}: {traceback.format_exc()}")
                    self.counter["clone_failed"] += 1
                    continue

                dataset += self._collect_repo(repo)
            except KeyboardInterrupt:
                raise
            except Exception as e:
                logger.error(f"Error processing {repo.full_name}: {e} {traceback.format_exc()}")
                self.counter["other_failed"] += 1
            finally:
                pbar.update()
                try:
                    su.io.rm(repo.dir)
                except Exception:
                    pass
        pbar.close()

        with_suffix = "" if suffix is None else f".{suffix}"
        su.io.dump(self.data_dir / f"coupjava-coarse{with_suffix}.jsonl", dataset, su.io.fmts.json_list)
        su.io.dump(self.data_dir / f"coupjava-coarse_counter{with_suffix}.json", self.counter, su.io.fmts.json_pretty)

    def _collect_repo(self, repo: Project) -> List[JavaVersionData]:
        dataset: List[JavaVersionData] = []

        # make sure to go back to the latest commit to get accurate history
        su.bash.run("git checkout -f origin/HEAD", 0, cwd=repo.dir)
        intervals: List[JavaVersionInterval] = self._search_java_version_intervals(repo.dir)
        logger.debug(f"{intervals=}")
        if len(intervals) == 0:
            self.counter["no_interval"] += 1
            return dataset
        elif len(intervals) == 1:
            self.counter["one_interval"] += 1
            return dataset

        # a tuple of last interval's ending (commit, timestamp, version)
        last_info = None

        for interval in intervals:
            if interval.version not in ALL_JAVA_VERSIONS:
                logger.warning(f"Not considering Java version {interval.version}")
                continue

            self.java_env_manager.switch(interval.version)
            commits = su.bash.run(
                f"git log --first-parent --pretty=%H {interval.beg_commit}..{interval.end_commit}", 0, cwd=repo.dir
            ).stdout.splitlines()
            commits.reverse()
            commits.insert(0, interval.beg_commit)
            logger.info(f"{repo.full_name}: interval version {interval.version} | {len(commits)} commits")

            # find compilable beginning commit
            beg_commit = None
            beg_commit_idx = self._maven_compile_flexible_commits(repo, commits, threshold=3, timeout=600)
            if beg_commit_idx is not None:
                beg_commit = commits[beg_commit_idx]
            else:
                self.counter["compilation_failed"] += 1

            # find compilable ending commit
            end_commit = None
            end_commit_idx = self._maven_compile_flexible_commits(
                repo, list(reversed(commits)), threshold=3, timeout=600
            )
            if end_commit_idx is not None:
                end_commit = commits[len(commits) - end_commit_idx - 1]
                new_last_info = (
                    end_commit,
                    su.bash.run(f"git show -s --format=%ct {end_commit}", 0, cwd=repo.dir).stdout.strip(),
                    interval.version,
                )
            else:
                self.counter["compilation_failed"] += 1
                new_last_info = None

            if last_info is not None:
                if interval.version < last_info[2]:
                    self.counter["downgrade_version"] += 1
                elif beg_commit is not None:
                    dataset.append(
                        JavaVersionData(
                            repo_name=repo.full_name,
                            old_commit=last_info[0],
                            old_timestamp=last_info[1],
                            old_version=last_info[2],
                            new_commit=beg_commit,
                            new_timestamp=su.bash.run(
                                f"git show -s --format=%ct {beg_commit}", 0, cwd=repo.dir
                            ).stdout.strip(),
                            new_version=interval.version,
                            patch=su.bash.run(f"git diff {last_info[0]} {beg_commit}", 0, cwd=repo.dir).stdout.strip(),
                        )
                    )

            last_info = new_last_info

        return dataset

    def _load_repos(self) -> List[Project]:
        repo_urls = su.io.load(self.repos_path)
        repos = [Project.from_github_url(url) for url in repo_urls]
        logger.debug(f"Loaded {len(repos)} repos")
        return repos

    RE_MAVEN_COMPILER_SOURCE_LINE = re.compile(
        r"\+.*<maven\.compiler\.source>(?P<version>\d+(\.\d+)?)</maven\.compiler\.source>"
    )

    def _search_java_version_intervals(self, repo_dir: su.arg.RPath) -> List[JavaVersionInterval]:
        """
        Searches for the commit intervals of each java version in the repo.
        """
        intervals: List[JavaVersionInterval] = []

        with su.io.cd(repo_dir):
            git_log = su.bash.run(
                "git log --first-parent -p -G '<maven.compiler.source>' --reverse --pretty=medium -- pom.xml", 0
            ).stdout

            last_version = None
            commit = None
            for line in git_log.splitlines():
                if line.startswith("commit "):
                    commit = line.split()[1]
                    continue
                match = self.RE_MAVEN_COMPILER_SOURCE_LINE.search(line)
                if match is not None:
                    version = normalize_java_version(match.group("version"))
                    # ignore formatting changes at the version line
                    if version != last_version:
                        # fill in the end commit of last interval
                        if len(intervals) > 0:
                            intervals[-1].end_commit = su.bash.run(f"git rev-parse {commit}^", 0).stdout.strip()
                        intervals.append(JavaVersionInterval(version, commit, None))
                        last_version = version
            if len(intervals) > 0:
                intervals[-1].end_commit = su.bash.run("git rev-parse HEAD", 0).stdout.strip()
        return intervals

    def _maven_compile_flexible_commits(
        self,
        repo: Project,
        commits: List[str],
        threshold: int = 5,
        timeout: int = 600,
    ) -> Optional[int]:
        """
        Try to maven compile the repo at the given commit, and if it fails, move down the commits list until succeed.
        The repo will be left at the last tried commit (if the return value is not None, that will be the commit where
        maven compile succeeds).
        "maven compile" means the `mvn clean test-compile` command.
        :param repo: the Maven repo being compiled.
        :param commits: the commits to try.
        :param threshold: the max number of commits to try.
        :param timeout: the max number of seconds to wait for the maven compile to succeed; if exceeded, return None.
        :return: the index of the commit where compilation succeeds, or None if no compilation succeeds.
        """
        threshold = min(threshold, len(commits))
        for i in range(threshold):
            commit = commits[i]
            logger.debug(f"{repo.full_name}: trying maven compile at {commit}")
            su.bash.run(f"git checkout -f {commit}", 0, cwd=repo.dir)
            try:
                with su.TimeUtils.time_limit(timeout):
                    rr = su.bash.run("mvn clean test-compile", cwd=repo.dir)
            except su.TimeoutException:
                logger.info(f"{repo.full_name}: maven compile timed out at {commit}")
                return None
            if rr.returncode == 0:
                return i
            logger.debug(f"{repo.full_name}: maven compile failed at {commit}, {rr.stdout}")
        logger.info(f"{repo.full_name}: maven compile failed after {threshold} tries: {commits[:threshold]}")
        logger.debug(f"mvn version: {su.bash.run('mvn -v', cwd=repo.dir).stdout}")
        return None

    def merge_batches(self, suffixes: List[str]):
        counter = collections.Counter()
        dataset = []
        for suffix in suffixes:
            counter_batch = su.io.load(self.data_dir / f"coupjava-coarse_counter.{suffix}.json")
            dataset_batch = su.io.load(self.data_dir / f"coupjava-coarse.{suffix}.jsonl")
            counter.update(counter_batch)
            dataset += dataset_batch
        su.io.dump(self.data_dir / "coupjava-coarse_counter.json", counter, su.io.fmts.json_pretty)
        su.io.dump(self.data_dir / "coupjava-coarse.jsonl", dataset, su.io.fmts.json_list)

    def filter_bad_patchset(self):
        old_dataset = su.io.load(self.data_dir / "coupjava-coarse.jsonl", clz=JavaVersionData)
        su.io.dump(self.data_dir / "coupjava-coarse.back.jsonl", old_dataset, su.io.fmts.json_list)
        new_dataset = []
        for d in old_dataset:
            try:
                unidiff.PatchSet(d.patch, metadata_only=True)
            except unidiff.errors.UnidiffParseError:
                continue

            if len(d.patch) == 0:
                continue

            new_dataset.append(d)
        su.io.dump(self.data_dir / "coupjava-coarse_dataset.jsonl", new_dataset, su.io.fmts.json_list)


if __name__ == "__main__":
    su.log.setup(level_stderr=su.log.WARNING, log_file=Path.cwd() / "debug.log", level_file=su.log.DEBUG)
    CLI(JavaVersionDatasetCollector, as_positional=False)
