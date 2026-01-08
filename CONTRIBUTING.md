# Contributing

Thanks for taking the time to contribute!

## Quick start

1. Fork the repo and create a feature branch.
2. Create/activate a virtualenv.
3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Run locally:

```bash
panel serve etf_bot.ipynb --show
```

## What to contribute

- Bug fixes (AkShare schema changes, UI issues, prompt formatting)
- Documentation improvements
- Small refactors that improve readability without changing behavior

## Style

- Keep changes minimal and focused.
- Prefer clear names and small helper functions.
- Avoid adding heavy dependencies unless necessary.

## Security / Secrets

Never commit API keys or any sensitive information.
Use environment variables (e.g. `LLM_API_KEY`) for local testing.

## Submitting a PR

- Describe what changed and why.
- Include steps to reproduce/verify.
- If the change affects user-facing behavior, update README.
