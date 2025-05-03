# CoUpJava

CoUpJava is the first large-scale dataset for code upgrade, focusing on the changes related to Java programming language's evolution. Quoting the abstract of [our MSR-DataTool'25 paper](https://pengyunie.github.io/p/JiangETAL25CoUpJava.pdf):

> Modern programming languages are constantly evolving, introducing new language features and APIs to enhance software development practices. Software developers often face the tedious task of upgrading their codebase to new programming language versions. Recently, large language models (LLMs) have demonstrated potential in automating various code generation and editing tasks, suggesting their applicability in automating code upgrade. However, there exists no benchmark for evaluating the code upgrade ability of LLMs, as distilling code changes related to programming language evolution from real-world software repositories’ commit histories is a complex challenge.
> In this work, we introduce CoUpJava, the first large-scale dataset for code upgrade, focusing on the code changes related to the evolution of Java. CoUpJava comprises 10,697 code upgrade samples, distilled from the commit histories of 1,379 open-source Java repositories and covering Java versions 7–23. The dataset is divided into two subsets: CoUpJava-Fine, which captures fine-grained method-level refactorings towards new language features; and CoUpJava-Coarse, which includes coarse-grained repository-level changes encompassing new language features, standard library APIs, and build configurations. Our proposed dataset provides high-quality samples by filtering irrelevant and noisy changes and verifying the compilability of upgraded code. Moreover, CoUpJava reveals diversity in code upgrade scenarios, ranging from small, fine-grained refactorings to large-scale repository modifications.

This repository contains our dataset and the scripts for collecting the dataset.

Note: the data files and some helper files required for running the scripts are too large to be uploaded to GitHub repository. They are available for download from [Zenodo][zenodo].

## Table of Contents

- [Dataset](#dataset)
- [Scripts](#scripts)
- [Citation](#citation)

## Dataset

CoUpJava consists of two parts. Download links are available on [Zenodo][zenodo].

- CoUpJava-Fine (`coupjava-fine.jsonl`), method-level code refactorings related to new language features in Java, collected with the help of RefactoringMiner

- CoUpJava-Coarse (`coupjava-coarse.jsonl`), repository-level upgrades from old Java versions to new ones, where the old and new versions of the repository are verified to be compilable

## Scripts

### Pre-requisites

#### Operating System

Linux is recommended. MacOS may work but was not tested.

#### Python

We suggest using [Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install) to prepare the environment for running our scripts. If you have `conda` installed, here is a quick command for automatically setting up the environment:

```
./prepare-env.sh
conda activate codeupgrade  # to activate the conda environment
```

#### Java

Our scripts will need JDK 7-23 and the corresponding versions Maven to attempt to build the repositories. We have packaged those JDKs and Maven binaries for your convenience.

Please download `javaenv.tar.gz` from [Zenodo][zenodo], and extract it to the `_work` directory of this repository (so that the JDKs are under `_work/javaenv/jdk` and Maven binaries are under `_work/javaenv/maven`). Our scripts will automatically use these files while running.

#### RefactoringMiner

Please install [RefactoringMiner](https://github.com/tsantalis/RefactoringMiner) and make sure the command `RefactoringMiner` is available in the system path.

### Collect list of repositories

Skip this step if you want to use the same list of repositories as ours (for reproducibility). The pre-collected list of repositories is available at `_work/repos/`.

Run the following commands to collect the list of GitHub repositories; note that this will likely lead to a different set of repositories compared to ours, because GitHub search results may change.
```
python -m codeupgrade.search_github --token <your_github_token> collect_repo
python -m codeupgrade.search_github combine_raw_repo
python -m codeupgrade.search_github remove_fork_combined_file
```

### Collect CoUpJava-Fine dataset

Run the following commands to collect the CoUpJava-Fine dataset.
```
# run RefactoringMiner to collect refactoring data
python -m codeupgrade.get_refactoring collect_refactoring

# build dataset
python -m codeupgrade.build_dataset_refactoringminer build_dataset_single_process

# filter dataset
python -m codeupgrade.build_dataset_refactoringminer build_dataset_filtered

# add commit_hash timestamp to dataset
python -m codeupgrade.build_dataset_refactoringminer build_dataset_with_timestamp

# remove invalid commit(timestamp)
python -m codeupgrade.build_dataset_refactoringminer remove_invalid_timestamp

# remove duplicate
python -m codeupgrade.build_dataset_refactoringminer remove_duplicate

# reformat dataset
python -m codeupgrade.build_dataset_refactoringminer reformat_dataset
```

### Collect CoUpJava-Coarse dataset

Run the following command to collect the CoUpJava-Coarse dataset.
```
python -m codeupgrade.build_dataset_javaver collect_repos
```

## Citation

If you use CoUpJava in your work, please cite the following paper:

```
@inproceedings{JiangETAL25CoUpJava,
    title={CoUpJava: A Dataset of Code Upgrade Histories in Open-Source Java Repositories},
    author={Jiang, Kaihang and Jin, Bihui and Nie, Pengyu},
    booktitle={International Conference on Mining Software Repositories, Data and Tool Showcase Track},
    year={2025},
}
```


---

[zenodo]: https://zenodo.org/records/15293313
