# Contributing

Changes to the operator, the contact-token definition or the readout configuration
change published numbers. Keep them separate from refactoring, and state which
artifacts are regenerated.

- Run `python -m pytest` before opening a pull request; the suite must pass on CPU.
- Run `ruff check .` and `black --check .` if you have the development extras
  installed.
- Keep the public interface in `src/pgl/__init__.py` compatible: additions are
  fine, renames need a changelog entry.
- Do not commit structures, affinity labels or model weights; they are obtained
  from their providers and stored outside the repository.
- Comments and docstrings use plain English, present tense, no marketing terms.
