# AbstractGraph Ecosystem

The AbstractGraph stack is split across four sibling repositories:

- `abstractgraph`
  Path: `/home/fabrizio/work/abstractgraph`
  Role: core representation, operators, XML, hashing, vectorization, display,
  compatibility shims, and graph adapters

- `abstractgraph-graphicalizer`
  Path: `/home/fabrizio/work/abstractgraph-graphicalizer`
  Role: raw-data-to-NetworkX graphicalizers, including attention-driven
  base-graph induction and chemistry conversion/drawing

- `abstractgraph-ml`
  Path: `/home/fabrizio/work/abstractgraph-ml`
  Role: estimators, neural models, feasibility, importance, and top-k analysis

- `abstractgraph-generative`
  Path: `/home/fabrizio/work/abstractgraph-generative`
  Role: rewriting, autoregressive and conditional generation, interpolation,
  optimization/repair, and story-graph tooling

Dependency direction:

- `abstractgraph`
- `abstractgraph-graphicalizer` depends on no sibling repos
- `abstractgraph-ml` depends on `abstractgraph`
- `abstractgraph-generative` depends on `abstractgraph` and `abstractgraph-ml`

Editable install order:

```bash
python -m pip install -e /home/fabrizio/work/abstractgraph --no-deps
python -m pip install -e /home/fabrizio/work/abstractgraph-graphicalizer --no-deps
python -m pip install -e /home/fabrizio/work/abstractgraph-ml --no-deps
python -m pip install -e /home/fabrizio/work/abstractgraph-generative --no-deps
```

## GitHub Workflow

The ecosystem is split across separate Git repositories, so syncing work means
committing and pushing each repo independently.

Typical sequence:

```bash
git -C /home/fabrizio/work/abstractgraph status
git -C /home/fabrizio/work/abstractgraph add -A
git -C /home/fabrizio/work/abstractgraph commit -m "Your message"
git -C /home/fabrizio/work/abstractgraph push origin main

git -C /home/fabrizio/work/abstractgraph-ml status
git -C /home/fabrizio/work/abstractgraph-ml add -A
git -C /home/fabrizio/work/abstractgraph-ml commit -m "Your message"
git -C /home/fabrizio/work/abstractgraph-ml push origin main

git -C /home/fabrizio/work/abstractgraph-generative status
git -C /home/fabrizio/work/abstractgraph-generative add -A
git -C /home/fabrizio/work/abstractgraph-generative commit -m "Your message"
git -C /home/fabrizio/work/abstractgraph-generative push origin main
```

Recommended remotes:

- `https://github.com/fabriziocosta/abstractgraph.git`
- `https://github.com/fabriziocosta/abstractgraph-ml.git`
- `https://github.com/fabriziocosta/abstractgraph-generative.git`

Check them with:

```bash
git -C /home/fabrizio/work/abstractgraph remote -v
git -C /home/fabrizio/work/abstractgraph-ml remote -v
git -C /home/fabrizio/work/abstractgraph-generative remote -v
```

## Credentials

This ecosystem has been used from multiple machines, and GitHub auth is not
uniform across them.

What to assume:

- do not assume `gh` is installed
- do not assume SSH auth works
- prefer clean HTTPS remotes
- if HTTPS push fails, it is usually a local credential-helper issue, not a repo issue

If `git push` fails with a credential-store error, inspect the machine-specific
reference in
[GITHUB_ENVIRONMENT_REFERENCE.md](/home/fabrizio/sync/Projects/AbstractGraphEcosystem/GITHUB_ENVIRONMENT_REFERENCE.md).

Two useful checks:

```bash
git config --global --get credential.helper
printf 'protocol=https\nhost=github.com\n\n' | git credential fill
```

If this machine has no usable GitHub credential helper configured, do not try
to work around it by storing tokens in repo config or committed files. Commit
locally and sync later from a machine with a working credential setup.

## Repo-Sync Rule

When one change spans multiple repos:

- commit each repo separately
- keep commit messages repo-specific
- push each repo separately
- verify `git status` is clean in all touched repos afterward

This is especially important for coordinated terminology or API migrations,
where `abstractgraph`, `abstractgraph-ml`, and `abstractgraph-generative`
must remain compatible at the same time.
