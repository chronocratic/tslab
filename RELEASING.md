# Releasing

Versions come from git tags (`hatch-vcs`): the tag `v0.1.0a1` builds version
`0.1.0a1`. Release notes are assembled from the fragments in `changelog.d/`.

## One-time setup

Do these once, in this order.

### GitHub

1. **Branches.** Create `dev` from `main` and push it. `main` stays the
   default branch.
2. **Rulesets.**
   - `main`: block force pushes and deletion, require a pull request, require
     linear history. Add the owner of `FF_MERGE_TOKEN` to the bypass list.
   - `dev`: block force pushes and deletion, require linear history, require
     the `CI` checks to pass.
3. **Fast-forward token.** Create a fine-grained personal access token for this
   repository with *Contents: read and write* and *Pull requests: read and
   write*. Create an environment `ff_merge` and add the token there as the
   secret `FF_MERGE_TOKEN`. The `/fast-forward` and `Release prep` workflows
   use it, because the default `GITHUB_TOKEN` cannot push to `main` past the
   ruleset or open pull requests.
4. **Labels.** Create `skip-changelog` and `changelog:added`,
   `changelog:changed`, `changelog:deprecated`, `changelog:removed`,
   `changelog:fixed`, `changelog:security`.
5. **PyPI environment.** Create an environment `pypi`. Optionally require a
   reviewer, so that each upload waits for approval.

### PyPI

Add a trusted publisher on PyPI (for a new project: *Your projects → Publishing
→ Add a new pending publisher*):

| Field | Value |
|-------|-------|
| PyPI project name | `tslab` |
| Owner | `Chronocratic` |
| Repository | `tslab` |
| Workflow | `pypi-publish.yml` |
| Environment | `pypi` |

No API token is stored anywhere.

### Read the Docs

1. Import the repository on [readthedocs.org](https://readthedocs.org) as the
   project `tslab`. The build reads `.readthedocs.yaml`.
2. **Check the connection.** *Admin → Integrations* must list a GitHub
   integration, and *Settings → GitHub Apps* (or *Webhooks*) of the repository
   must show Read the Docs. Without it, Read the Docs never hears about new
   tags or pushes. Push a commit to `dev` and check that a build starts.
3. *Admin → Settings*: default branch `main` (the `latest` version), enable
   *Build pull requests for this project*.
4. *Admin → Automation rules*, for tags matching the SemVer versions pattern:
   - *Activate version*;
   - *Set version as default*, so the newest pre-release is what readers see
     while there is no final release yet.
5. From `v0.1.0` on, delete the *Set version as default* rule and set the
   default version to `stable`, which Read the Docs points at the newest
   final release.

## Each release

1. **Changelog.** Run the `Release prep` workflow (*Actions → Release prep →
   Run workflow*) with the version without the `v`, for example `0.1.0a1`. It
   builds `CHANGELOG.md` from the fragments on `dev`, removes the fragments,
   and opens a pull request into `dev`. Review the notes and merge it.
2. **Main.** Open a pull request from `dev` to `main`. When CI passes, comment
   `/fast-forward`. The workflow backs up `main`, fast-forwards it, and keeps
   the five newest backups.
3. **Release.** Create a GitHub release from `main` with a new tag `v<version>`,
   for example `v0.1.0a1`. Tick *Set as a pre-release* for alpha, beta and
   release-candidate versions. Publishing the release:
   - builds and uploads the package to PyPI (`Publish to PyPI`);
   - replaces the release notes with the matching `CHANGELOG.md` section
     (`Sync release notes`);
   - makes Read the Docs build and activate the docs of the tag.
4. **Check.** `uv pip install tslab==<version>` in a fresh environment, and
   open the docs of the tag on Read the Docs.

## If something goes wrong

- **The PyPI upload failed**: fix the cause, then re-run the failed
  `Publish to PyPI` job. A version can never be uploaded twice, so if a broken
  package was uploaded, yank it on PyPI and release the next version.
- **The docs of a tag are missing**: check the Read the Docs connection
  (one-time setup, step 2), then *Versions → Add version* or re-sync versions.
- **The fast-forward failed**: `dev` is not ahead of `main`. Rebase `dev` on
  `main`; the ruleset of `dev` blocks the force push this needs, so disable it
  for the push and enable it again right after. Then comment `/fast-forward`
  again.
