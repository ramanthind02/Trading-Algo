<!-- source: https://nautilustrader.io/docs/latest/getting_started/installation -->

[Installation](#installation)

NautilusTrader is officially supported for Python 3.12-3.14 on the following 64-bit platforms:

| Operating System | Supported Versions | CPU Architecture |
|---|---|---|
| Linux (Ubuntu) | 22.04 and later | x86_64 |
| Linux (Ubuntu) | 22.04 and later | ARM64 |
| macOS | 15.0 and later | ARM64 |
| Windows Server | 2022 and later | x86_64 |

NautilusTrader may work on other platforms, but only those listed above are regularly used by developers and tested in CI.

Continuous CI coverage comes from the GitHub Actions runners we build on:

`Linux (Ubuntu)`

builds currently pin to`ubuntu-22.04`

to keep glibc 2.35 compatibility even as`ubuntu-latest`

moves ahead.`macOS (ARM64)`

builds run on`macos-latest`

, so support tracks that runner image as it moves ahead.`Windows (x86_64)`

builds currently pin to`windows-2022`

to keep the toolchain stable.

On Linux, confirm your glibc version with `ldd --version`

and ensure it reports 2.35 or newer before proceeding.

We recommend using the latest supported version of Python and installing [nautilus_trader](https://pypi.org/project/nautilus_trader/) inside a virtual environment to isolate dependencies.

**There are two supported ways to install**:

- Pre-built binary wheel from PyPI
*or*the Nautech Systems package index. - Build from source.

We highly recommend installing using the [uv](https://docs.astral.sh/uv) package manager with a "vanilla" CPython.

Conda and other Python distributions *may* work but aren’t officially supported.

[From PyPI](#from-pypi)

To install the latest [nautilus_trader](https://pypi.org/project/nautilus_trader/) binary wheel (or sdist package) from PyPI:

[Extras](#extras)

Install optional dependencies as 'extras' for specific integrations:

`betfair`

: Betfair adapter (integration) dependencies.`docker`

: Needed for Docker when using the IB gateway (with the Interactive Brokers adapter).`ib`

: Interactive Brokers adapter (integration) dependencies.`polymarket`

: Polymarket adapter (integration) dependencies.`visualization`

: Plotly-based interactive tearsheets and charts.

To install with specific extras:

[From the Nautech Systems package index](#from-the-nautech-systems-package-index)

The Nautech Systems package index (`packages.nautechsystems.io`

) complies with [PEP-503](https://peps.python.org/pep-0503/) and hosts both stable and development binary wheels for `nautilus_trader`

.
This enables users to install either the latest stable release or pre-release versions for testing.

[Stable wheels](#stable-wheels)

Stable wheels correspond to official releases of `nautilus_trader`

on PyPI, and use standard versioning.

To install the latest stable release:

Use `--extra-index-url`

instead of `--index-url`

if you want uv to fall back to PyPI automatically:

[Development wheels](#development-wheels)

Development wheels are published from both the `nightly`

and `develop`

branches,
allowing users to test features and fixes ahead of stable releases.

This process also helps preserve compute resources and provides easy access to the exact binaries tested in CI pipelines,
while adhering to [PEP-440](https://peps.python.org/pep-0440/) versioning standards:

`develop`

wheels use the version format`dev{date}+{build_number}`

(e.g.,`1.208.0.dev20241212+7001`

).`nightly`

wheels use the version format`a{date}`

(alpha) (e.g.,`1.208.0a20241212`

).

| Platform | Nightly | Develop |
|---|---|---|
`Linux (x86_64)` | ✓ | ✓ |
`Linux (ARM64)` | ✓ | - |
`macOS (ARM64)` | ✓ | - |
`Windows (x86_64)` | ✓ | - |

**Note**: Development wheels from the `develop`

branch publish for Linux x86_64 only.
Windows, macOS, and Linux ARM64 builds run on the nightly schedule to keep CI feedback fast.

We do not recommend using development wheels in production environments, such as live trading controlling real capital.

[Installation commands](#installation-commands)

By default, uv will install the latest stable release. Adding the `--pre`

flag ensures that pre-release versions, including development wheels, are considered.

To install the latest available pre-release (including development wheels):

To install a specific development wheel (e.g., `1.221.0a20250912`

for September 12, 2025):

[Available versions](#available-versions)

You can view all available versions of `nautilus_trader`

on the [package index](https://packages.nautechsystems.io/simple/nautilus-trader/index.html).

To programmatically request and list available versions:

[Branch updates](#branch-updates)

`develop`

branch wheels (`.dev`

): Build and publish continuously with every merged commit.`nightly`

branch wheels (`a`

): Build and publish daily when we automatically merge the`develop`

branch at**14 UTC**(if there are changes).

[Retention policies](#retention-policies)

`develop`

branch wheels (`.dev`

): We retain only the most recent wheel build.`nightly`

branch wheels (`a`

): We retain only the 30 most recent wheel builds.

[Verifying build provenance](#verifying-build-provenance)

All release artifacts published by the project carry cryptographic attestations generated by the CI/CD pipeline:

- Python wheels and source distribution (PyPI, GitHub Releases, Nautech Systems package index):
[SLSA](https://slsa.dev/)build provenance. - Docker images (
`ghcr.io/nautechsystems/nautilus_trader`

,`ghcr.io/nautechsystems/jupyterlab`

): keyless[cosign](https://github.com/sigstore/cosign)signatures plus SPDX SBOM attestations.

Both are issued via [Sigstore](https://www.sigstore.dev/) and bound to a specific
commit SHA, so verification ensures the artifact was produced by the official
NautilusTrader GitHub Actions workflow and has not been tampered with since.

For step-by-step verification commands, see [Verifying releases](https://github.com/nautechsystems/nautilus_trader/blob/develop/SECURITY.md#verifying-releases) in `SECURITY.md`

.

Verification requires the [GitHub CLI](https://cli.github.com/) (`gh`

) for Python artifacts
and [cosign](https://github.com/sigstore/cosign) for Docker images.
Development wheels from `develop`

and `nightly`

branches are also attested.

[From source](#from-source)

It's possible to install from source using pip if you first install the build dependencies as specified in the `pyproject.toml`

.

[Install clang](#3-install-clang)

Install [clang](https://clang.llvm.org/) (a C language frontend for LLVM). On Linux this also installs [lld](https://lld.llvm.org/), which is configured as the Rust linker for faster builds:

Verify: `clang --version`


[Clone and install](#5-clone-and-install)

Clone the source with `git`

, and install from the project's root directory:

The `--depth 1`

flag fetches just the latest commit for a faster, lightweight clone.

[Install Cap'n Proto for development](#6-install-capn-proto-for-development)

Install [Cap'n Proto](https://capnproto.org/) if you plan to enable the `capnp`

Rust feature,
regenerate serialization schemas, or work on serialization code. Use the repository script on
Linux or macOS to install the pinned version from `tools.toml`

:

Verify: `capnp --version`


Cap'n Proto is a development dependency. It is not required when installing pre-built wheels.

[Set environment variables](#7-set-environment-variables)

Set environment variables for PyO3 compilation (Linux and macOS only). Run these commands from
the repository root after `uv sync`

:

The `LD_LIBRARY_PATH`

export is Linux-specific and not needed on macOS.

The `PYTHONHOME`

variable is required when running `make cargo-test`

with a `uv`

-installed Python.
Without it, tests that depend on PyO3 may fail to locate the Python runtime.

[From GitHub release](#from-github-release)

To install a binary wheel from GitHub, first navigate to the [latest release](https://github.com/nautechsystems/nautilus_trader/releases/latest).
Download the appropriate `.whl`

for your operating system and Python version, then run:

[Versioning and releases](#versioning-and-releases)

NautilusTrader is still under active development. Some features may be incomplete, and while
the API is becoming more stable, breaking changes can occur between releases.
We strive to document these changes in the release notes on a **best-effort basis**.

We aim to follow a **bi-weekly release schedule**, though experimental or larger features may cause delays.

Use NautilusTrader only if you are prepared to adapt to these changes.

[Redis](#redis)

Using [Redis](https://redis.io) with NautilusTrader is **optional** and only required if configured as the backend for a cache database or [message bus](../../concepts/message_bus).

The minimum supported Redis version is 6.2 (required for [streams](https://redis.io/docs/latest/develop/data-types/streams/) functionality).

For a quick setup, we recommend using a [Redis Docker container](https://hub.docker.com/_/redis/). You can find an example setup in the `.docker`

directory,
or run the following command to start a container:

This command will:

- Pull the latest version of Redis from Docker Hub if it's not already downloaded.
- Run the container in detached mode (
`-d`

). - Name the container
`redis`

for easy reference. - Expose Redis on the default port 6379, making it accessible to NautilusTrader on your machine.

To manage the Redis container:

- Start it with
`docker start redis`

- Stop it with
`docker stop redis`


We recommend using [Redis Insight](https://redis.io/insight/) as a GUI to visualize and debug Redis data efficiently.

[Precision mode](#precision-mode)

NautilusTrader supports two precision modes for its core value types (`Price`

, `Quantity`

, `Money`

),
which differ in their internal bit-width and maximum decimal precision.

**High-precision**: 128-bit integers with up to 16 decimals of precision, and a larger value range.**Standard-precision**: 64-bit integers with up to 9 decimals of precision, and a smaller value range.

By default, the official Python wheels ship in high-precision (128-bit) mode on Linux and macOS.
On Windows, only standard-precision (64-bit) Python wheels are available because MSVC's C/C++ frontend
does not support `__int128`

, preventing the Cython/FFI layer from handling 128-bit integers.

For pure Rust crates, high-precision works on all platforms (including Windows) since Rust handles
`i128`

/`u128`

via software emulation. The default is standard-precision unless you explicitly enable
the `high-precision`

feature flag.

The performance tradeoff is that standard-precision is ~3–5% faster in typical backtests, but has lower decimal precision and a smaller representable value range.

Performance benchmarks comparing the modes are pending.

[Build configuration](#build-configuration)

The precision mode is determined by:

- Setting the
`HIGH_PRECISION`

environment variable during compilation,**and/or** - Enabling the
`high-precision`

Rust feature flag explicitly.

[Rust feature flag](#rust-feature-flag)

To enable high-precision (128-bit) mode in Rust, add the `high-precision`

feature to your `Cargo.toml`

:

See the [Value Types](../../concepts/overview#value-types) specifications for more details.
