# Documentation source

MkDocs + Material + mkdocstrings, in the style of effector's docs.

```
make docs-serve     # live preview at http://127.0.0.1:8000
make docs-build     # strict build into documentation/site/ (git-ignored)
make docs-assets    # rerun the examples: figures + printed output
```

- `mkdocs.yml` — config and the four top tabs.
- `docs/` — the pages. Each top tab is a hand-written hub page
  (`guides.md`, `examples.md`, `api_docs.md`) linking to the pages below it.
- `docs/static/` — figures, written by `make_assets.py`.
- `_outputs/` — the examples' printed output (git-ignored). Paste from here
  into the pages' `text` blocks, so no page shows a number the package did
  not print.

Page skeletons. Guide: `title` front matter, `???+ success "Description"`,
`???+ note "Reading time"`, declarative `##` sections, `## Where to next`.
API page: `## Summary`, `---`, `## Usage`, `## API` with `### ::: path`.

`../docs/` is **not** this: it holds the landing page. `.github/workflows/pages.yml`
publishes both on every push to `main`: the landing page at `/calm-additive/`,
these docs under `/calm-additive/docs/`.
