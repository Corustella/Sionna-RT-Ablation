# Third-party notices

`RTBackend.lattice_slice` in `rt_backend.py` adapts the Fibonacci lattice
formula from NVIDIA Sionna RT v1.1.0, `src/sionna/rt/utils/ray_tracing.py`.
The adaptation adds a global index offset and total-point denominator so
successive bounded-memory calls evaluate disjoint parts of the original
lattice. The rest of the upstream ray-tracing implementation is used through
the installed package, not copied into this repository.

Source: https://github.com/NVlabs/sionna-rt/blob/v1.1.0/src/sionna/rt/utils/ray_tracing.py

SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
