# Contributing to qullama

## Pull requests

- Branch from `master`; keep the diff minimal
- Run the tests and the commands your change affects before opening the PR


## AI policy

AI-assisted contributions are welcome - but you are responsible for what you submit.
AI generated comments and PR are not allowed, a person should be in the middle of the process.


## Releasing (maintainers only)

Releases are tag-driven; `.github/workflows/release.yml` does the work:

1. Bump `version` in `pyproject.toml` if needed and commit
2. Tag and push: `git tag v0.2.0 && git push origin v0.2.0`.


## License

By contributing you agree that your contributions are licensed under the
[MIT License](LICENSE), same as the rest of the project.
