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

import os
import sys


def sort_by_dict_order(file_path):
    with open(file_path, "r") as f:
        lines = f.readlines()
    sorted_lines = sorted(lines)
    with open(file_path, "w") as f:
        f.writelines(sorted_lines)


if __name__ == "__main__":
    file_paths = sys.argv[1:]
    for file_path in file_paths:
        file_path = os.path.normpath(file_path)
        sort_by_dict_order(file_path)
