# Contributing

1. Create a focused branch and keep generated outputs out of Git.
2. Do not commit portraits, registries, model weights, API keys, or local
   absolute paths.
3. Keep the numbered pipeline deterministic and document threshold changes.
4. Add a small synthetic or redistributable example when changing image logic.
5. Run `python tmuscle.py doctor` and
   `python -m compileall -q final tools src tmuscle` before opening a pull
   request.

For scoring changes, include before/after score tables and the exact formula,
weights, rounding, and clipping behaviour.
