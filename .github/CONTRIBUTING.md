# Contributing to SimpleOptions

Thank you for taking the time to consider contributing to SimpleOptions! Please take a quick look through these guidelines to help ensure we can merge your contribution as quickly as possible!

## Testing

Testing, by and large, is a good thing. It helps ensure the code in this repository is correct, and prevents regressions from occurring. This is particularly important given that a growing number scientific publications rely on this repository.

Ideally, all critical areas of code should be tested for correctness. For example:

- If you're implementing an equation or algorithm, test that it produces correct values for a range of inputs.
- If you're implementing an agent, run it for a number of time steps on a dummy problem and ensure that learned values (e.g., a policy or value function) are correct.

Please use `PyTest` to write your tests, and add them to the `test` subdirectory.
All tests are automatically run for every push and pull request. Your code must pass all tests before it can be merged to master.

You are free to organise your tests as you see fit. For isntance, you use individual functions for each test, or group related tests together in a class.

## Code Style

You should format your code using the [Black](https://black.readthedocs.io/en/stable/) formatter.
When using Black to format your code pass in `--line-length 120` and leave all other settings at their default values.
Both regular Python (.py) and Jupyter notebook (.ipynb) files should be formatted.

You can check whether your code is formatted correctly by checking that our Black formatting GitHub action runs successfully.
If it fails, check the action's logs to see which file is the culprit, and fix it (usually by running Black) before trying again.

## Versioning

SimpleOptions uses semantic versioning. We haven't always done so in the past, but we are doing so now. When you make a pull request, how you change the version number depends on how your pull request changes the public API (i.e., the classes, methods, and functions that developers will interact with when they use SimpleOptions).

If your pull request includes a non-backward-compatible change to the public API, you should increment the major release number (e.g., `0.8.3` → `1.0.0`).

If your pull request includes a backward compatible change to the public API, you should increment the minor release number (e.g., `0.8.3` → `0.9.0`).

If your pull request only includes minor changes that are backward compatible and do not not change the public API in any way, you should increment the patch number (e.g., `0.8.3` → `0.8.4`).

If in doubt, ask for the version number to be changed for you when you create your pull request.
