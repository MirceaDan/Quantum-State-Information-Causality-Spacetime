# Relational Dynamics Toy Model

Reference implementation of the attached computational ansatz. Components are explicitly tagged as established physics, mathematical definitions, modeling assumptions, speculative extensions, or numerical approximations.

The project follows the staged development order in the specification. Tests are written before the full implementation and are the executable contract.

## Development

```powershell
python -m pip install -e ".[dev]"
pytest
```

The external simulation clock, when used by future optimizers, is a numerical control variable only. It is not stored as a physical degree of freedom.
