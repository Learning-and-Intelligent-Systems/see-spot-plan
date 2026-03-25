# see-spot-plan
A collection of tools to connect a Spot robot with vision models and planning utilities. The goal is to be simple, minimal, and easy-to-use.

## Installation
* This repository uses Python versions 3.10-3.11. We recommend 3.10.14.
* Create a new conda environment with `conda create --name see_spot_plan python=3.10.14`
* Run `pip install -e .` to install dependencies. This also generates required protobuf files. If you need to regenerate them manually, run `python generate_proto.py`.


## Development

Install dev dependencies (linter + type checker):
```
pip install -e ".[develop]"
```

### Linting
```
ruff check
```

### Type checking
```
ty check .
```

If you're using conda locally, pass the Python path explicitly:
```
ty check --python $(which python) .
```

Both tools are written in Rust and run in under a second on this repo.

## Contributing

* Keep changes small and focused — one feature or fix per PR.
* Run `ruff check` and `ty check .` before pushing. CI runs both on every push.
* Follow existing code style (enforced by ruff via `ruff.toml`).
* Add docstrings to new public functions (enforced by the `D1` ruff rule).

## Running/examples
Currently, the main script in the repo is `spot_localization.py`. Here is an example command (for the LIS spot Jasper):
```
python spot_utils/spot_localization.py --hostname 192.168.80.3 --map_name b45-621
```
You can 'hijack' the robot via the tablet when prompted, move it around, and then have it print out its pose! We can then command it to navigate to any saved pose via other utilities (which are forthcoming).
