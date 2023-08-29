"""Static code checking.

Run formatter, linter and type checker.
"""

import argparse
import subprocess
import sys

from app.core.settings import Settings


def colored_result(return_code: int) -> str:
    """Return a result displayed with an ANSI color based on return_code.

    Check ANSI colors here: https://i.stack.imgur.com/9UVnC.png

    Args:
        return_code: 0 if it was a success, rest if concidered a failure.

    Returns:
        green "Success" of red "Failure"
    """
    if return_code == 0:
        return "\033[32;1mSuccess\033[0m"  # Green

    return "\033[31;1mFailure\033[0m"  # Red


def main(args: argparse.Namespace) -> int:
    """Run formatter, linter and type checker.

    Args:
        args: provided by user.

    Returns:
        0 if there was no issue, >0 otherwise.
    """
    settings = Settings()
    padding_right = 20

    print("Black formatter")
    print("".ljust(padding_right, "-"))
    black_args = ["black", "."]
    if args.fix is not True:
        black_args.append("--check")
    black_result = subprocess.run(black_args, check=False)
    print("")

    print("Ruff linter")
    print("".ljust(padding_right, "-"))
    ruff_args = [
        "ruff",
        "check",
        ".",
    ]
    if args.fix is True:
        ruff_args.append("--fix")
    if settings.IS_GH_ACTION is True:
        ruff_args.append("--format=github")
    ruff_result = subprocess.run(
        ruff_args,
        check=False,
    )
    if ruff_result.returncode == 0:
        print(colored_result(ruff_result.returncode))
    print("")

    print("Mypy type checker")
    print("".ljust(padding_right, "-"))
    mypy_result = subprocess.run(["mypy", "."], check=False)
    print("")

    print("Results")
    print(
        "- Formatter".ljust(padding_right, "."),
        colored_result(black_result.returncode),
    )
    print(
        "- Linter".ljust(padding_right, "."),
        colored_result(ruff_result.returncode),
    )
    print(
        "- Type checker".ljust(padding_right, "."),
        colored_result(mypy_result.returncode),
    )

    return black_result.returncode + ruff_result.returncode + mypy_result.returncode


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Run static code analysis tools, which are: "
            "formatter, linter and static type checker.\n\n"
        )
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Format files with Black and autofix lintinig issues with Ruff.",
    )
    sys.exit(main(parser.parse_args()))
