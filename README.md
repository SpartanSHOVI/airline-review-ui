# Airline Review UI

This repository contains a Streamlit-based Airline Review Dashboard UI. The UI was designed and iterated using Pencil MCP and exported assets.

What's included
- Streamlit app: `app.py`
- Styles: `styles/streamlit_theme.css`
- Processed background: `prod/assets/processed_bg.jpg`
- Placeholder production template: `prod/index.html`
- Image processing script: `scripts/process_background.py`

How to run locally
1. Create a Python 3.10+ virtual environment and activate it:

```bash
python -m venv .venv
.\.venv\Scripts\activate   # Windows
source .venv/bin/activate   # macOS / Linux
```

2. Install dependencies:

```bash
pip install -r requirements.txt
pip install matplotlib pillow
```

3. Start Streamlit:

```bash
streamlit run app.py
```

Pushing to GitHub
- This repo is not connected to a remote yet. To push, create a GitHub repository named `airline-review-ui` and follow the standard `git remote add` and `git push` commands.
