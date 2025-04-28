import collections
import os
from pathlib import Path
from typing import Optional

# we study java versions from 7 to 23
ALL_JAVA_VERSIONS = list(range(7, 24))


def normalize_java_version(version: str) -> int:
    """
    Normalize the two styles of java version (1.major.minor... and major.minor...) to a single number representing
    the major version.
    """
    if version.startswith("1."):
        return int(version.split(".")[1])
    else:
        return int(version.split(".")[0])


class JavaEnvManager:
    """
    Manages multiple Java and Maven versions.
    """

    def __init__(self, env_dir: Path):
        self.env_dir = env_dir
        self.orig_path = os.environ["PATH"]
        self.cur_version: Optional[int] = None

    JDKS = {
        7: "jdk1.7.0_80",
        8: "jdk1.8.0_411",
        9: "jdk-9.0.4",
        10: "jdk-10.0.2",
        11: "jdk-11.0.23",
        12: "jdk-12.0.2",
        13: "jdk-13.0.2",
        14: "jdk-14.0.2",
        15: "jdk-15.0.2",
        16: "jdk-16.0.2",
        17: "jdk-17.0.11",
        18: "jdk-18.0.2.1",
        19: "jdk-19.0.2",
        20: "jdk-20.0.2",
        21: "jdk-21.0.3",
        22: "jdk-22.0.1",
        23: "jdk-23.0.1",
    }

    JDK_2_MAVEN = collections.defaultdict(
        lambda: "3.9.9",
        {
            7: "3.8.8",
        },
    )

    def switch(self, version: int):
        """
        Switches to the given Java version.

        Setup the JAVA_HOME and PATH environment variables to use the given version of
        JDK and the compatible version of Maven.
        """
        jdk_path = self.env_dir / "jdk" / self.JDKS[version]
        maven_version = self.JDK_2_MAVEN[version]
        maven_bin_path = self.env_dir / "maven" / f"apache-maven-{maven_version}" / "bin"
        os.environ["JAVA_HOME"] = str(jdk_path.absolute())
        os.environ["PATH"] = f"{jdk_path.absolute()}/bin:{maven_bin_path}:{self.orig_path}"
        self.cur_version = version
