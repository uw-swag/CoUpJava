# CoUpJava

CoUpJava is the first large-scale dataset for code upgrade, focusing on the changes related to Java programming language's evolution. Quoting the abstract of [our MSR-DataTool'25 paper](https://pengyunie.github.io/p/JiangETAL25CoUpJava.pdf):

> Modern programming languages are constantly evolving, introducing new language features and APIs to enhance software development practices. Software developers often face the tedious task of upgrading their codebase to new programming language versions. Recently, large language models (LLMs) have demonstrated potential in automating various code generation and editing tasks, suggesting their applicability in automating code upgrade. However, there exists no benchmark for evaluating the code upgrade ability of LLMs, as distilling code changes related to programming language evolution from real-world software repositories’ commit histories is a complex challenge.
> In this work, we introduce CoUpJava, the first large-scale dataset for code upgrade, focusing on the code changes related to the evolution of Java. CoUpJava comprises 10,697 code upgrade samples, distilled from the commit histories of 1,379 open-source Java repositories and covering Java versions 7–23. The dataset is divided into two subsets: CoUpJava-Fine, which captures fine-grained method-level refactorings towards new language features; and CoUpJava-Coarse, which includes coarse-grained repository-level changes encompassing new language features, standard library APIs, and build configurations. Our proposed dataset provides high-quality samples by filtering irrelevant and noisy changes and verifying the compilability of upgraded code. Moreover, CoUpJava reveals diversity in code upgrade scenarios, ranging from small, fine-grained refactorings to large-scale repository modifications.


Here you can find the data files and some helper files required for running the scripts. The scripts for collecting the dataset are available on [our GitHub repository](https://github.com/uw-swag/CoUpJava).


If you use CoUpJava in your work, please cite the following paper:

```
@inproceedings{JiangETAL25CoUpJava,
    title={CoUpJava: A Dataset of Code Upgrade Histories in Open-Source Java Repositories},
    author={Jiang, Kaihang and Jin, Bihui and Nie, Pengyu},
    booktitle={International Conference on Mining Software Repositories, Data and Tool Showcase Track},
    year={2025},
}
```
