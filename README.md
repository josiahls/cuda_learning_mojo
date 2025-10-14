# C Binder Mojo
> A generalized c beinding API for mojo.

> Note: Mojo is a very new and changing langauge. Any and all libs being developed for it so far will be esentially Alpha builds. This means they very likely will break between versions. This project I am using for mujoco c binding. If you would like to use this for your project, I welcome contributions.

> Additionally: This repo is verions 0.0.0.pre-alpha. For the time being this is largely unfriendly to use.

# Installation

Minimal installation requires pixi:
```bash
curl -fsSL https://pixi.sh/install.sh | sh
```

Below command will pull all the deps, build them, and run the tests to verify the installation worked.
```bash
pixi run test_all
```

# Usage
> This is a super early alpha stage right now very likely will not work for usecases beyond what I can develop against (mujoco / opencv). Here be dragons.
> With that said, here are the core commands to generate a binded c project. 

Reference `tests/test_c_project` for a working example. This should be populated after running `pixi run test_all`.

```bash
pixi run generate_bindings --help
```

```bash
pixi run package --help
```

```bash
pixi run configure
```