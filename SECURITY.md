# Security and credential handling

## Immediate action for a tracked credential file

A public repository must not contain `kaggle.json`, personal access tokens, cloud service keys, or notebook cells that embed credentials.

If `kaggle.json` or any token-bearing file has ever been committed:

1. **Revoke or rotate the credential immediately.** Removing a file from the latest commit does not invalidate a credential that may already have been copied.
2. Remove it from the current index:

   ```bash
   git rm --cached --ignore-unmatch kaggle.json
   git add .gitignore
   git commit -m "security: remove credential file and harden ignores"
   ```

3. Remove it from Git history if it was present in earlier commits. With `git-filter-repo` installed:

   ```bash
   git filter-repo --path kaggle.json --invert-paths
   git push --force-with-lease --all
   git push --force-with-lease --tags
   ```

   Coordinate before rewriting history if anyone else has cloned or forked the repository.

4. Check notebooks, logs, issues, pull requests, releases, and GitHub Actions output for copied credentials.
5. Enable GitHub secret scanning and push protection where available.

Do **not** place the contents of the exposed file into an issue, commit message, patch, or security report.

## Safe local and Colab usage

Use one of these mechanisms instead of tracked files:

- Colab Secrets (`google.colab.userdata`)
- environment variables
- a local file outside the repository with restrictive permissions
- the platform's official credential store

Example environment-variable pattern:

```bash
export KAGGLE_USERNAME='...'
export KAGGLE_KEY='...'
```

For a local Kaggle configuration file, keep it at `~/.kaggle/kaggle.json`, not inside the repository, and restrict permissions:

```bash
chmod 600 ~/.kaggle/kaggle.json
```

## GitHub authentication

Use the GitHub CLI credential store, SSH keys, or a platform-managed token. Do not paste a `ghp_...` token into a notebook, README, setup script, or shell command that may be saved in history.

## Reporting a vulnerability

Report suspected credential exposure privately to the repository owner. Include the affected path and commit range, but never reproduce the secret itself.
