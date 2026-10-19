"""
Build and deploy the transcript worker Lambda (youtube-transcript-processor).

Builds a clean package from lambda-transcript-processor/requirements.txt plus
lambda_function.py, never from the folder's loose contents, so stray files
(test_local.py, old vendored libraries, __pycache__) are not shipped.
Dependencies are installed as Linux wheels for the function's own runtime and
architecture, read from AWS, so a Windows machine builds what Lambda runs.

Deploying uploads the zip, publishes an immutable version described as
YYYYMMDD_<name> (<git sha>), and points the `prod` alias at it. The API invokes
`prod`, so rolling back is pointing the alias at the previous version; the
script prints that command.

Dry run by default: builds the zip and prints what it would do. Pass --apply
to deploy. Uses the default AWS credentials.

    poetry run python scripts/deploy_lambda.py --name request_timeout
    poetry run python scripts/deploy_lambda.py --name request_timeout --apply
"""

import argparse
import datetime
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import boto3

FUNCTION_NAME = "youtube-transcript-processor"
ALIAS = "prod"
REGION = "us-east-2"

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "lambda-transcript-processor"
BUILD_DIR = SOURCE_DIR / ".build"
PLATFORMS = {"x86_64": "manylinux2014_x86_64", "arm64": "manylinux2014_aarch64"}


def git_revision() -> str:
    sha = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=ROOT, capture_output=True, text=True, check=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--", str(SOURCE_DIR)],
        cwd=ROOT, capture_output=True, text=True, check=True,
    ).stdout.strip()
    return f"{sha}+dirty" if dirty else sha


def build_zip(python_version: str, architecture: str) -> Path:
    package_dir = BUILD_DIR / "package"
    zip_path = BUILD_DIR / "lambda.zip"
    shutil.rmtree(BUILD_DIR, ignore_errors=True)
    package_dir.mkdir(parents=True)

    subprocess.run(
        [
            sys.executable, "-m", "pip", "install",
            "-r", str(SOURCE_DIR / "requirements.txt"),
            "--target", str(package_dir),
            "--platform", PLATFORMS[architecture],
            "--implementation", "cp",
            "--python-version", python_version,
            "--only-binary=:all:",
            "--no-compile",
            "--quiet",
        ],
        check=True,
    )
    shutil.copy2(SOURCE_DIR / "lambda_function.py", package_dir)

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(package_dir.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(package_dir).as_posix())
    return zip_path


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--name", required=True, help="short change name for the version description")
    parser.add_argument("--apply", action="store_true", help="deploy and move the prod alias")
    args = parser.parse_args()

    client = boto3.client("lambda", region_name=REGION)
    config = client.get_function_configuration(FunctionName=FUNCTION_NAME)
    python_version = config["Runtime"].removeprefix("python")
    architecture = config["Architectures"][0]
    current_version = client.get_alias(FunctionName=FUNCTION_NAME, Name=ALIAS)["FunctionVersion"]

    zip_path = build_zip(python_version, architecture)
    description = f"{datetime.date.today():%Y%m%d}_{args.name} ({git_revision()})"
    print(f"Built {zip_path} ({zip_path.stat().st_size / 1e6:.1f} MB) for python{python_version}/{architecture}")
    print(f"{ALIAS} currently points to version {current_version}")

    if not args.apply:
        print(f"Dry run: would publish '{description}' and point {ALIAS} to it. Pass --apply to deploy.")
        return

    response = client.update_function_code(
        FunctionName=FUNCTION_NAME, ZipFile=zip_path.read_bytes()
    )
    client.get_waiter("function_updated_v2").wait(FunctionName=FUNCTION_NAME)
    version = client.publish_version(
        FunctionName=FUNCTION_NAME,
        CodeSha256=response["CodeSha256"],
        Description=description,
    )["Version"]
    client.update_alias(FunctionName=FUNCTION_NAME, Name=ALIAS, FunctionVersion=version)

    print(f"Published version {version} '{description}'; {ALIAS} -> {version}")
    print(
        f"Roll back: aws lambda update-alias --function-name {FUNCTION_NAME} "
        f"--name {ALIAS} --function-version {current_version} --region {REGION}"
    )


if __name__ == "__main__":
    main()
