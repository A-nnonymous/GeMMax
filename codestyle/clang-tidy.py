# ruff: noqa
#!/usr/bin/env python3

# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#
# ===- run-clang-tidy.py - Parallel clang-tidy runner ---------*- python -*--===#
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# ===------------------------------------------------------------------------===#
# FIXME: Integrate with clang-tidy-diff.py

"""
Parallel clang-tidy runner
==========================

Runs clang-tidy over all files in a compilation database. Requires clang-tidy
and clang-apply-replacements in $PATH.

Example invocations.
- Run clang-tidy on all files in the current working directory with a default
  set of checks and show warnings in the cpp files and all project headers.
    run-clang-tidy.py $PWD

- Fix all header guards.
    run-clang-tidy.py -fix -checks=-*,llvm-header-guard

- Fix all header guards included from clang-tidy and header guards
  for clang-tidy headers.
    run-clang-tidy.py -fix -checks=-*,llvm-header-guard extra/clang-tidy \
                      -header-filter=extra/clang-tidy

Compilation database setup:
http://clang.llvm.org/docs/HowToSetupToolingForLLVM.html
"""


import argparse
import glob
import json
import multiprocessing
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import traceback

try:
    import yaml
except ImportError:
    yaml = None

is_py2 = sys.version[0] == "2"

if is_py2:
    import Queue as queue
else:
    import queue


def find_compilation_database(path, result="./"):
    """Adjusts the directory until a compilation database is found."""
    result = "./"
    while not os.path.isfile(os.path.join(result, path)):
        if os.path.realpath(result) == "/":
            print("Warning: could not find compilation database.")
            return None
        result += "../"
    return os.path.realpath(result)


def make_absolute(f, directory):
    """Convert a relative file path to an absolute file path."""
    if os.path.isabs(f):
        return f
    return os.path.normpath(os.path.join(directory, f))


def analysis_gitignore(path, filename=".gitignore"):
    """Analysis gitignore file and return ignore file list"""
    with open(path + "/" + filename, "r") as f:
        lines = f.readlines()
        ignore_file_list = []
        for line in lines:
            # Blank row
            if line == "\n" or line == "\r\n":
                continue

            # explanatory note
            line = line.replace("\n", "").strip()
            if "#" in line:
                if not line.startswith("#"):
                    ignore_file_list.append(line[: line.index("#")].replace(" ", ""))
                continue

            # TODO(gouzil): support more gitignore rules
            if "*" in line:
                continue

            ignore_file_list.append(line.replace(" ", ""))

    return ignore_file_list


def skip_check_file(database, build_path):
    """Skip checking some files"""
    skip_file_list = []
    skip_file_list.append(".cu")
    skip_file_list.append(os.path.join(os.getcwd(), build_path))
    skip_file_list += analysis_gitignore(os.getcwd())
    res_list = []
    for entry in database:
        write_in = True
        for ignore_file in skip_file_list:
            if ignore_file in entry["file"]:
                write_in = False
                break
        if write_in:
            res_list.append(entry)

    return res_list


def get_tidy_invocation(
    f,
    clang_tidy_binary,
    checks,
    tmpdir,
    build_path,
    header_filter,
    extra_arg,
    extra_arg_before,
    quiet,
    config,
):
    """Gets a command line for clang-tidy."""
    start = [clang_tidy_binary]
    if header_filter is not None:
        start.append("-header-filter=" + header_filter)
    if checks:
        start.append("-checks=" + checks)
    if tmpdir is not None:
        start.append("-export-fixes")
        # Get a temporary file. We immediately close the handle so clang-tidy can
        # overwrite it.
        (handle, name) = tempfile.mkstemp(suffix=".yaml", dir=tmpdir)
        os.close(handle)
        start.append(name)
    for arg in extra_arg:
        start.append(f"-extra-arg={arg}")
    for arg in extra_arg_before:
        start.append(f"-extra-arg-before={arg}")
    start.append("-p=" + build_path)
    if quiet:
        start.append("-quiet")
    if config:
        start.append("-config=" + config)
    start.append(f)
    return start


def merge_replacement_files(tmpdir, mergefile):
    """Merge all replacement files in a directory into a single file"""
    # The fixes suggested by clang-tidy >= 4.0.0 are given under
    # the top level key 'Diagnostics' in the output yaml files
    mergekey = "Diagnostics"
    merged = []
    for replacefile in glob.iglob(os.path.join(tmpdir, "*.yaml")):
        content = yaml.safe_load(open(replacefile, "r"))
        if not content:
            continue  # Skip empty files.
        merged.extend(content.get(mergekey, []))

    if merged:
        # MainSourceFile: The key is required by the definition inside
        # include/clang/Tooling/ReplacementsYaml.h, but the value
        # is actually never used inside clang-apply-replacements,
        # so we set it to '' here.
        output = {"MainSourceFile": "", mergekey: merged}
        with open(mergefile, "w") as out:
            yaml.safe_dump(output, out)
    else:
        # Empty the file:
        open(mergefile, "w").close()


def check_clang_apply_replacements_binary(args):
    """Checks if invoking supplied clang-apply-replacements binary works."""
    try:
        subprocess.check_call([args.clang_apply_replacements_binary, "--version"])
    except:
        print(
            "Unable to run clang-apply-replacements. Is clang-apply-replacements "
            "binary correctly specified?",
            file=sys.stderr,
        )
        traceback.print_exc()
        sys.exit(1)


def apply_fixes(args, tmpdir):
    """Calls clang-apply-fixes on a given directory."""
    invocation = [args.clang_apply_replacements_binary]
    if args.format:
        invocation.append("-format")
    if args.style:
        invocation.append("-style=" + args.style)
    invocation.append(tmpdir)
    subprocess.call(invocation)


def run_tidy(args, tmpdir, build_path, queue, lock, failed_files):
    """Takes filenames out of queue and runs clang-tidy on them."""
    while True:
        name = queue.get()
        invocation = get_tidy_invocation(
            name,
            args.clang_tidy_binary,
            args.checks,
            tmpdir,
            build_path,
            args.header_filter,
            args.extra_arg,
            args.extra_arg_before,
            args.quiet,
            args.config,
        )

        proc = subprocess.Popen(
            invocation, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        output, err = proc.communicate()
        if proc.returncode != 0:
            failed_files.append(name)
        with lock:
            sys.stdout.write(" ".join(invocation) + "\n" + output.decode("utf-8"))
            if len(err) > 0:
                sys.stdout.flush()
                sys.stderr.write(err.decode("utf-8"))
        queue.task_done()


def main():
    """Runs clang-tidy over all files in a compilation database."""
    parser = argparse.ArgumentParser(
        description="Runs clang-tidy over all files "
        "in a compilation database. Requires "
        "clang-tidy and clang-apply-replacements in "
        "$PATH."
    )
    parser.add_argument(
        "-clang-tidy-binary",
        metavar="PATH",
        default="clang-tidy-15",
        help="path to clang-tidy binary",
    )
    parser.add_argument(
        "-clang-apply-replacements-binary",
        metavar="PATH",
        default="clang-apply-replacements-15",
        help="path to clang-apply-replacements binary",
    )
    parser.add_argument(
        "-checks",
        default=None,
        help="checks filter, when not specified, use clang-tidy " "default",
    )
    parser.add_argument(
        "-config",
        default=None,
        help="Specifies a configuration in YAML/JSON format: "
        "  -config=\"{Checks: '*', "
        "                       CheckOptions: [{key: x, "
        '                                       value: y}]}" '
        "When the value is empty, clang-tidy will "
        "attempt to find a file named .clang-tidy for "
        "each source file in its parent directories.",
    )
    parser.add_argument(
        "-header-filter",
        default=None,
        help="regular expression matching the names of the "
        "headers to output diagnostics from. Diagnostics from "
        "the main file of each translation unit are always "
        "displayed.",
    )
    if yaml:
        parser.add_argument(
            "-export-fixes",
            metavar="filename",
            dest="export_fixes",
            help="Create a yaml file to store suggested fixes in, "
            "which can be applied with clang-apply-replacements.",
        )
    parser.add_argument(
        "-j",
        type=int,
        default=0,
        help="number of tidy instances to be run in parallel.",
    )
    parser.add_argument(
        "files",
        nargs="*",
        default=[".*"],
        help="files to be processed (regex on path)",
    )
    parser.add_argument("-fix", action="store_true", help="apply fix-its")
    parser.add_argument(
        "-format",
        action="store_true",
        help="Reformat code " "after applying fixes",
    )
    parser.add_argument(
        "-style",
        default="file",
        help="The style of reformat " "code after applying fixes",
    )
    parser.add_argument(
        "-p",
        dest="build_path",
        help="Path used to read a compile command database.",
    )
    parser.add_argument(
        "-extra-arg",
        dest="extra_arg",
        action="append",
        default=[],
        help="Additional argument to append to the compiler " "command line.",
    )
    parser.add_argument(
        "-extra-arg-before",
        dest="extra_arg_before",
        action="append",
        default=[],
        help="Additional argument to prepend to the compiler " "command line.",
    )
    parser.add_argument(
        "-quiet", action="store_true", help="Run clang-tidy in quiet mode"
    )
    args = parser.parse_args()

    db_path = "compile_commands.json"

    if args.build_path is not None:
        build_path = args.build_path
        if not os.path.isfile(os.path.join(build_path, db_path)):
            print(
                f"Warning: could not find compilation database in {build_path}, skip clang-tidy check."
            )
            build_path = None
    else:
        # Find our database
        build_path = find_compilation_database(db_path)
    if build_path is None:
        sys.exit(0)

    try:
        invocation = [args.clang_tidy_binary, "-list-checks"]
        invocation.append("-p=" + build_path)
        if args.checks:
            invocation.append("-checks=" + args.checks)
        invocation.append("-")
        if args.quiet:
            # Even with -quiet we still want to check if we can call clang-tidy.
            with open(os.devnull, "w") as dev_null:
                subprocess.check_call(invocation, stdout=dev_null)
        else:
            subprocess.check_call(invocation)
    except:
        print("Unable to run clang-tidy.", file=sys.stderr)
        sys.exit(0)

    # Load the database and extract all files.
    database = json.load(open(os.path.join(build_path, db_path)))
    database = skip_check_file(database, build_path)
    files = {make_absolute(entry["file"], entry["directory"]) for entry in database}

    max_task = args.j
    if max_task == 0:
        max_task = multiprocessing.cpu_count()

    tmpdir = None
    if args.fix or (yaml and args.export_fixes):
        check_clang_apply_replacements_binary(args)
        tmpdir = tempfile.mkdtemp()

    # Build up a big regexy filter from all command line arguments.
    file_name_re = re.compile("|".join(args.files))

    return_code = 0
    try:
        # Spin up a bunch of tidy-launching threads.
        task_queue = queue.Queue(max_task)
        # List of files with a non-zero return code.
        failed_files = []
        lock = threading.Lock()
        for _ in range(max_task):
            t = threading.Thread(
                target=run_tidy,
                args=(args, tmpdir, build_path, task_queue, lock, failed_files),
            )
            t.daemon = True
            t.start()

        # Fill the queue with files.
        for name in files:
            if file_name_re.search(name):
                task_queue.put(name)

        # Wait for all threads to be done.
        task_queue.join()
        if len(failed_files):
            return_code = 1

    except KeyboardInterrupt:
        # This is a sad hack. Unfortunately subprocess goes
        # bonkers with ctrl-c and we start forking merrily.
        print("\nCtrl-C detected, goodbye.")
        if tmpdir:
            shutil.rmtree(tmpdir)
        os.kill(0, 9)

    if yaml and args.export_fixes:
        print("Writing fixes to " + args.export_fixes + " ...")
        try:
            merge_replacement_files(tmpdir, args.export_fixes)
        except:
            print("Error exporting fixes.\n", file=sys.stderr)
            traceback.print_exc()
            return_code = 1

    if args.fix:
        print("Applying fixes ...")
        try:
            apply_fixes(args, tmpdir)
        except:
            print("Error applying fixes.\n", file=sys.stderr)
            traceback.print_exc()
            return_code = 1

    if tmpdir:
        shutil.rmtree(tmpdir)
    sys.exit(return_code)


if __name__ == "__main__":
    if os.getenv("SKIP_CLANG_TIDY_CHECK", "").lower() in [
        "y",
        "yes",
        "t",
        "true",
        "on",
        "1",
    ]:
        print(
            "SKIP_CLANG_TIDY_CHECK is set, skip clang-tidy check.",
            file=sys.stderr,
        )
        sys.exit(0)

    target_version = "15.0.2"
    try:
        out = subprocess.check_output(["clang-tidy", "--version"], shell=True)
        version = out.decode("utf-8")
        if version.find(target_version) == -1:
            print(
                f"clang-tidy version == {target_version} not found, attempting auto-install...",
                file=sys.stderr,
            )
            subprocess.check_output(
                'pip install --no-cache clang-tidy=="15.0.2.1"',
                shell=True,
            )
    except:
        print("clang-tidy not found, attempting auto-install...", file=sys.stderr)
        subprocess.check_output(
            'pip install --no-cache clang-tidy=="15.0.2.1"',
            shell=True,
        )
    main()
