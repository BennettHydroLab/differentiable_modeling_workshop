"""Author a notebook from a plain-text source file.

Hand-writing .ipynb JSON is error-prone; this turns a readable source file
into a valid notebook. Split cells with a line containing only ``#%% md`` or
``#%% code``.

    uv run python tools/build_notebook.py src.txt notebooks/theory/01_foo.ipynb

Add ``#%% code tags=remove-output`` to attach MyST cell tags.
"""

import sys
from pathlib import Path

import nbformat as nbf


def parse(text):
    cells, kind, tags, buf = [], None, [], []

    def flush():
        if kind is None:
            return
        src = "\n".join(buf).strip("\n")
        if not src.strip():
            return
        cell = nbf.v4.new_markdown_cell(src) if kind == "md" else nbf.v4.new_code_cell(src)
        if tags:
            cell.metadata["tags"] = list(tags)
        cells.append(cell)

    for line in text.splitlines():
        if line.startswith("#%%"):
            flush()
            parts = line[3:].split()
            kind = parts[0] if parts else "code"
            tags = []
            for p in parts[1:]:
                if p.startswith("tags="):
                    tags = p[5:].split(",")
            buf = []
        else:
            buf.append(line)
    flush()
    return cells


def main():
    src, dest = Path(sys.argv[1]), Path(sys.argv[2])
    nb = nbf.v4.new_notebook(cells=parse(src.read_text()))
    nb.metadata.update({
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    })
    dest.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, str(dest))
    n_code = sum(1 for c in nb.cells if c.cell_type == "code")
    print(f"wrote {dest}  ({len(nb.cells)} cells, {n_code} code)")


if __name__ == "__main__":
    main()
