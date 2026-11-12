# Releasing

Versions come from git tags (`hatch-vcs`): the tag `v0.1.0a1` builds version
`0.1.0a1`. Release notes are assembled from the fragments in `changelog.d/`.

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
- **The docs of a tag are missing**: check that Read the Docs still receives GitHub
  webhooks (*Admin → Integrations*), then *Versions → Add version* or re-sync versions.
- **The fast-forward failed**: `dev` is not ahead of `main`. Rebase `dev` on
  `main`; the ruleset of `dev` blocks the force push this needs, so disable it
  for the push and enable it again right after. Then comment `/fast-forward`
  again.
