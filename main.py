from __future__ import annotations

import argparse
import json

# NOTE: These three modules can modify the system state and should be used with caution.
#       Confirm that the operations are safe before running them.
import os  # NOTE: not safe, verify usage
import pathlib
import platform
import shutil
import subprocess
import tempfile
import time
import typing
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from urllib.parse import urlencode


class ComputerInfo(typing.TypedDict):
    os: str
    os_version: str
    cpu: str
    commit: str
    timestamp: str
    cli_args: dict | None
    additional_info: str | None


@dataclass
class PerformanceMetric:
    operation_name: str
    duration_seconds: float
    timestamp: datetime
    additional_info: dict | None = None


def build_issue_content(*, results: dict[str, typing.Any], total_time: float) -> str:
    computer_info = get_computer_info()

    labels: list[str] = [computer_info["os"]]

    if total_time < 10:
        labels.append("fast")
    elif total_time < 100:
        labels.append("slow")
    else:
        labels.append("very-slow")

    separator = ";"
    title = f"{computer_info['os']} {separator} {computer_info['cpu']} {separator} {total_time:.2f}s {separator} {get_run_name()}"
    labels_str = ",".join(labels)
    body = f"""
```json
{json.dumps(results, indent=4)}
```
"""

    query_params = {
        "title": title,
        "body": body,
        "labels": labels_str,
    }
    return urlencode(query_params)


def get_link_for_issue(github_repo_url: str, issue_content: str) -> str:
    return f"{github_repo_url}/issues/new?{issue_content}"


def get_user_name() -> str:
    try:
        import pwd

        return pwd.getpwuid(os.getuid()).pw_name
    except ImportError:
        # For Windows systems
        return os.getlogin()


def get_git_version_string(repo_path: str = ".") -> str:
    try:
        # Get current commit hash (short version)
        commit_hash = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=repo_path, text=True
        ).strip()

        # Check if repository is dirty (has uncommitted changes)
        status_output = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=repo_path, text=True
        )
        is_dirty = len(status_output.strip()) > 0

        # Format the version string
        return f"{commit_hash}+{'dirty' if is_dirty else 'clean'}"

    except subprocess.CalledProcessError:
        return "not-from-git-repo"
    except FileNotFoundError:
        return "not-from-git-repo"


def get_computer_info(
    *,
    additional_info: str | None = None,
    args: argparse.Namespace | None = None,
) -> ComputerInfo:
    cli_args = vars(args) if args else None
    info: ComputerInfo = {
        "os": platform.system(),
        "os_version": platform.version(),
        "cpu": platform.processor(),
        "commit": get_git_version_string(str(pathlib.Path(__file__).parent)),
        "timestamp": get_run_name(),
        "cli_args": cli_args,
        "additional_info": additional_info,
    }
    return info


def print_computer_info(info: ComputerInfo):
    print("\nComputer Info:")
    for key, value in info.items():
        print(f"{key}: {value}")


@lru_cache(maxsize=128)
def get_run_name() -> str:
    return f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"


def get_results_file_name(custom_filename: str | None = None) -> str:
    if custom_filename:
        return custom_filename
    os_name = get_computer_info()["os"].lower()
    return f"{get_user_name()}_{os_name}_{get_run_name()}.json"


class PerformanceTest:
    def __init__(
        self,
        base_dir: str | None = None,
        title: str = "Performance Test",
        output_dir: str | None = None,
    ):
        self.title = title
        self.base_dir = base_dir or tempfile.mkdtemp()
        self.metrics: list[PerformanceMetric] = []
        self.results_directory: pathlib.Path = (
            pathlib.Path(output_dir) if output_dir else pathlib.Path(__file__).parent / "_results"
        )

    def get_total_time(self) -> float:
        return sum(metric.duration_seconds for metric in self.metrics)

    @staticmethod
    def measure_operation(operation_name: str):
        def decorator(func):
            def wrapper(self, *args, **kwargs):
                start_time = time.time()
                result = func(self, *args, **kwargs)
                end_time = time.time()

                metric = PerformanceMetric(
                    operation_name=operation_name,
                    duration_seconds=end_time - start_time,
                    timestamp=datetime.now(),
                )
                self.metrics.append(metric)
                return result

            return wrapper

        return decorator

    def print_results(self, total_time: float | None = None):
        total_time = total_time or self.get_total_time()

        print(f"# {self.title} Results:")
        print("-" * 50)

        combined = defaultdict(list)
        for metric in self.metrics:
            combined[metric.operation_name].append(metric)
        for operation_name, metrics in combined.items():
            suffix = f"{operation_name} (x{len(metrics)})".ljust(40)
            operation_time = sum(metric.duration_seconds for metric in metrics)

            # Create a visual bar showing percentage of total time
            bar_width = 100
            filled_chars = int((operation_time / total_time) * bar_width)
            bar = "█" * filled_chars + "░" * (bar_width - filled_chars)

            percentage_of_total = operation_time / total_time * 100

            print(
                f">>> {suffix}: {operation_time:.8f} seconds ({percentage_of_total:.2f}%)\n[{bar}]\n"
            )
        print("-" * 50)

    def get_results_dicts(self) -> list[dict]:
        combined = defaultdict(list)
        for metric in self.metrics:
            combined[metric.operation_name].append(metric)

        result = []
        for operation_name, metrics in combined.items():
            result.append(
                {
                    "operation_name": operation_name,
                    "duration_seconds": sum(metric.duration_seconds for metric in metrics),
                    "count": len(metrics),
                }
            )
        return result

    @classmethod
    def get_results_dict(
        cls, metrics_results: list[dict], computer_info: ComputerInfo
    ) -> dict[str, typing.Any]:
        for result in metrics_results:
            if "timestamp" in result and isinstance(result["timestamp"], datetime):
                result["timestamp"] = result["timestamp"].isoformat()

        return {
            "results": metrics_results,
            "computer_info": computer_info,
        }

    @classmethod
    def save_results_json(cls, results: dict[str, typing.Any]):
        # Convert datetime objects to strings

        performance_test = cls()

        path_to_file = pathlib.Path(
            performance_test.results_directory,
            get_results_file_name(),
        )

        path_to_file.parent.mkdir(parents=True, exist_ok=True)

        with path_to_file.open("w") as f:
            json.dump(
                results,
                f,
                indent=4,
            )

        print(f"\nYour results have been saved to: '{path_to_file}'")


class GitOperationsTest(PerformanceTest):
    def __init__(
        self,
        repo_url: str,
        base_dir: str | None = None,
        output_dir: str | None = None,
    ):
        super().__init__(title="Git Operations", base_dir=base_dir, output_dir=output_dir)
        self.repo_url = repo_url
        self.target_commit_hash = "f9514ac4b263be971c50f7e0f719b7a6d361e192"

    @property
    def repo_path(self) -> str:
        return os.path.join(self.base_dir, "repo")

    def setup(self):
        os.makedirs(self.repo_path, exist_ok=True)

    # NOTE: we are not measuring the time it takes to clone the repo - that's
    #       not something dependent on the performance of the machine. At least not only on that.
    def clone_repository(self):
        if os.path.exists(self.repo_path) and os.path.isdir(self.repo_path):
            # Check if it's a git repo and has the correct remote
            try:
                result = subprocess.run(
                    ["git", "remote", "get-url", "origin"],
                    cwd=self.repo_path,
                    capture_output=True,
                    text=True,
                    check=True,
                )
                if result.stdout.strip() == self.repo_url:
                    return
            except subprocess.CalledProcessError:
                # NOTE: Here script is removing cloned repository if it exists and then cloning
                # it again. This is done to make sure that the cloned repo is in a clean state.
                shutil.rmtree(self.repo_path)
        subprocess.run(["git", "clone", self.repo_url, self.repo_path], check=True)

    @PerformanceTest.measure_operation("Checkout 11 Commits")
    def checkout_commits(self):
        commits = [
            "cc4db3294aeec3256d9e7319b0aa534a34ed3db8",
            "d4e29ea3fef8008508169c6807440788ed7ae5f4",
            "ad268bb80e91d2da7ade632ec10c7e5888277165",
            "551c5613a9d70c9ec69e7119ee39a35d3dac0326",
            "4881d1e225445faa196f2de58c4ce02dd32f5837",
            "293d7c3bf887280a9111bac712c4a5360a8948ae",
            "bb8c2a64981d3e806575a445115f29eddf014c77",
            "406c092a3bf65bbd4405ce87611a7e0b9c0ae706",
            "4f8157588e47f909276e0474f6926740a2e55b9c",
            "f9514ac4b263be971c50f7e0f719b7a6d361e192",  # newest one
        ]
        for commit in commits:
            subprocess.run(["git", "checkout", commit], cwd=self.repo_path, check=True)
        subprocess.run(
            ["git", "checkout", self.target_commit_hash],
            cwd=self.repo_path,
            check=True,
        )

    @PerformanceTest.measure_operation("Revert Last 50 Commits")
    def revert_recent_commits(self, count: int = 50):
        os.chdir(self.repo_path)

        # Get last 50 commit hashes
        result = subprocess.run(
            ["git", "log", f"-{count}", "--format=%H"],
            capture_output=True,
            text=True,
            check=True,
        )
        commit_hashes = result.stdout.strip().split("\n")

        for commit_hash in commit_hashes:
            try:
                subprocess.run(["git", "revert", "--no-commit", commit_hash], check=True)
                subprocess.run(
                    ["git", "commit", "--no-verify", "-m", f'Revert "{commit_hash}"'], check=True
                )
            except subprocess.CalledProcessError:
                print(f"Failed to revert commit {commit_hash}, skipping...")
                subprocess.run(["git", "revert", "--abort"], check=False)
                continue

        subprocess.run(
            ["git", "checkout", self.target_commit_hash],
            cwd=self.repo_path,
            check=True,
        )

    def run(self):
        self.setup()
        self.clone_repository()
        self.checkout_commits()
        self.revert_recent_commits()


class FileOperationsTest(PerformanceTest):
    def __init__(self, base_dir: str | None = None, output_dir: str | None = None):
        super().__init__(title="File Operations", base_dir=base_dir, output_dir=output_dir)
        self._test_files: list[str] = []

    @PerformanceTest.measure_operation("Create 10000 Files")
    def create_multiple_files(self, count: int = 10000) -> list[str]:
        created_files = []
        for i in range(count):
            content = f"This is test file {i}\nIt contains some test content.\nLine 3 of the file.\nCreated at: {datetime.now()}"
            file_path = os.path.join(self.base_dir, f"test_file_{i}.txt")
            with open(file_path, "w") as f:
                f.write(content)
            created_files.append(file_path)
            if i % 100 == 0:
                print(f"Created {i} files...")
        self._test_files.extend(created_files)
        return created_files

    @PerformanceTest.measure_operation("Read File")
    def read_file(self, file_path: str) -> str:
        with open(file_path) as f:
            return f.read()

    @PerformanceTest.measure_operation("Modify File Content")
    def modify_file(self, file_path: str, old_text: str, new_text: str):
        content = self.read_file(file_path)
        modified_content = content.replace(old_text, new_text)
        with open(file_path, "w") as f:
            f.write(modified_content)

    @PerformanceTest.measure_operation("Cleanup - Delete all test files")
    def cleanup(self):
        """Clean up all created test files"""
        for file_path in self._test_files:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
            except Exception as e:
                print(f"Failed to remove {file_path}: {e}")
        self._test_files.clear()

    def run(self):
        files = self.create_multiple_files()
        for idx, file_path in enumerate(files):
            self.read_file(file_path)
            self.modify_file(file_path, "test", "performance_test")
            if idx % 100 == 0:
                print(f"Processed {idx} files...")
        self.cleanup()


class CPUTest(PerformanceTest):
    def __init__(self, base_dir: str | None = None, output_dir: str | None = None):
        super().__init__(title="CPU Operations", base_dir=base_dir, output_dir=output_dir)

    @PerformanceTest.measure_operation("Prime Numbers")
    def calculate_primes(self, up_to=100000):
        primes = []
        for num in range(2, up_to):
            if all(num % i != 0 for i in range(2, int(num**0.5) + 1)):
                primes.append(num)
        return primes

    def run(self):
        self.calculate_primes()


class MemoryTest(PerformanceTest):
    def __init__(self, base_dir: str | None = None, output_dir: str | None = None):
        super().__init__(title="Memory Operations", base_dir=base_dir, output_dir=output_dir)

    @PerformanceTest.measure_operation("Large Array Allocation")
    def allocate_large_array(self, size=10000000):
        return list(range(size))

    @PerformanceTest.measure_operation("Dictionary Operations")
    def dictionary_operations(self, size=1000000):
        d = {}
        for i in range(size):
            d[f"key_{i}"] = f"value_{i}"
        for i in range(size):
            _ = d[f"key_{i}"]

    def run(self):
        self.allocate_large_array()
        self.dictionary_operations()


def parse_args():
    parser = argparse.ArgumentParser(description="Run performance tests and save results")
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="./_results",
        help="Directory to save results (default: ./_results)",
    )
    parser.add_argument(
        "--output-file",
        "-f",
        type=str,
        help="Custom output file name (default: username_os_timestamp.json)",
    )
    parser.add_argument(
        "--additional-info",
        "-i",
        type=str,
        help="Additional information to include in results",
    )
    parser.add_argument(
        "--github-repo-url",
        type=str,
        help="Github repo url",
        default="https://github.com/piotrgredowski/pc-performance-test",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    base_dir = pathlib.Path(__file__).parent / "_cloned_repo"

    git_test = GitOperationsTest(
        repo_url="https://github.com/fastapi/fastapi",
        base_dir=str(base_dir),
        output_dir=args.output_dir,
    )
    git_test.run()

    file_test = FileOperationsTest(output_dir=args.output_dir)
    file_test.run()

    cpu_test = CPUTest(output_dir=args.output_dir)
    cpu_test.run()

    memory_test = MemoryTest(output_dir=args.output_dir)
    memory_test.run()

    total_time = (
        git_test.get_total_time()
        + file_test.get_total_time()
        + cpu_test.get_total_time()
        + memory_test.get_total_time()
    )

    git_test.print_results(total_time)
    file_test.print_results(total_time)
    cpu_test.print_results(total_time)
    memory_test.print_results(total_time)

    print(f"\nTotal time: {total_time:.8f} seconds")

    computer_info = get_computer_info(additional_info=args.additional_info, args=args)
    print_computer_info(computer_info)

    all_operations_results = [
        {"operation_name": "Total Time", "duration_seconds": total_time},
        *git_test.get_results_dicts(),
        *file_test.get_results_dicts(),
        *cpu_test.get_results_dicts(),
        *memory_test.get_results_dicts(),
    ]

    results_dict = PerformanceTest.get_results_dict(
        metrics_results=all_operations_results, computer_info=computer_info
    )

    PerformanceTest.save_results_json(results_dict)

    issue_content = build_issue_content(results=results_dict, total_time=total_time)
    issue_url = get_link_for_issue(
        github_repo_url=args.github_repo_url,
        issue_content=issue_content,
    )

    print(
        "\n" + "-" * 50,
        "\nYou can create an Github issue with your results by opening below link in your browser:",
        "\n" + "-" * 50,
        f"\n{issue_url}",
        "\n" + "-" * 50,
    )


if __name__ == "__main__":
    main()
