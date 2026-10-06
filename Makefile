.PHONY: test docs-serve docs-build docs-assets

test:
	uv run --group test pytest tests -q

# live preview at http://127.0.0.1:8000
docs-serve:
	uv run --group docs mkdocs serve -f documentation/mkdocs.yml

docs-build:
	uv run --group docs mkdocs build -f documentation/mkdocs.yml --strict

# rerun the examples' code: figures -> documentation/docs/static,
# printed output -> documentation/_outputs (paste into the pages)
docs-assets:
	uv run python documentation/make_assets.py
