# doc2poma

[![PyPI version](https://img.shields.io/pypi/v/doc2poma.svg)](https://pypi.org/project/doc2poma/)  
[![License: MPL-2.0](https://img.shields.io/badge/License-MPL%202.0-brightgreen.svg)](LICENSE)

> **Part of the POMA toolkit.** For the complete documentation, see the [organization README](https://github.com/poma-science/.github).

Convert PDFs, HTML, images, and Markdown into a standardized `.poma` archive.

---

## ⚙️ Core Idea

`doc2poma` normalizes any document into:

- **`content.md`**: one-sentence-per-line structured Markdown
- **`assets/`**: extracted images, page screenshots
- **`tables/`**: extracted tables

Metadata for the archive is stored directly within the ZIP container's metadata storage, eliminating the need for a separate manifest.json file.

Uses poma-senter at the end of the conversion flow.

> Everything you need for hierarchy-preserving chunking.

---

## 🚀 Installation

```bash
pip install doc2poma
```

Note: For SVG processing, you need to have at least one of the following tools installed:
- Inkscape
- CairoSVG (requires libcairo)
- Wand (requires ImageMagick)
- svglib + reportlab

---

## 🏁 Quick Start

```python
from doc2poma import convert

config = {
    "conversion_provider": "openai",
    "conversion_model": "gpt-4o",
}

# turn any file into a .poma archive
archive, costs = convert(full_file_path="doc.pdf", config=config, base_url= None)
print(archive)

# Note: convert() will raise ValueError if model_config is missing or invalid
# and FileNotFoundError if the input file doesn't exist
```

---

## 🔗 Next Steps

Once you have `*.poma`, feed it into **poma-chunker**:

```bash
pip install poma-chunker
```

See the full POMA workflow in the org-level README:
🔗 [Module Matrix](https://github.com/poma-science/.github#module-matrix)

---

## 🛠 Tests

```bash
pytest
```

---

## 🤝 Contribute

1. Fork & clone
2. `pip install -e .`
3. Add tests & docs
4. PR!

---

## 📜 License

MPL-2.0 — see [LICENSE](LICENSE)

---

## 🧑‍🔬 Acknowledgments

This package uses the following open source libraries:

- [poma-senter](https://github.com/poma-science/poma-senter) (MPL-2.0)
- [litellm](https://github.com/BerriAI/litellm) (MIT)
- [importlib](https://docs.python.org/3/library/importlib.html) (Python Software Foundation License)
- [pandas](https://github.com/pandas-dev/pandas) (BSD-3-Clause)
- [Pillow](https://github.com/python-pillow/Pillow) (HPND)
- [pillow-heif](https://github.com/carsales/pyheif) (MIT)
- [html2text](https://github.com/Alir3z4/html2text) (GPL-3.0)
- [beautifulsoup4](https://www.crummy.com/software/BeautifulSoup/) (MIT)
- [pymupdf](https://github.com/pymupdf/PyMuPDF) (AGPL-3.0)
- [pytest](https://github.com/pytest-dev/pytest) (MIT)
- [cairosvg](https://github.com/Kozea/CairoSVG) (LGPL-3.0)
- [pycairo](https://github.com/pygobject/pycairo) (LGPL-2.1)
- [rlPyCairo](https://github.com/pmaupin/rlPyCairo) (MIT)
- [svglib](https://github.com/deeplook/svglib) (LGPL-3.0)
- [reportlab](https://github.com/rl-project/reportlab) (BSD-3-Clause)
- [wand](https://github.com/emcconville/wand) (MIT)

See each linked repository for full license terms.
Some dependencies are licensed under AGPL-3.0 or GPL-3.0; these are used only at runtime for certain file conversions or rendering. If you re-distribute a bundled version of this package with these libraries included, ensure compliance with their respective licenses.
