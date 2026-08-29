# Documentation

This branch (`gh-pages`) hosts the GitHub Pages site for `bit-compact`.

- Site entry: `index.html` (served from branch root)
- Styles: `style.css`
- No Jekyll processing: `.nojekyll`

The main library lives on `main` — cloning `main` does **not** pull this branch's files:

```bash
git clone https://github.com/TheElephantCoder/bit-compact.git # gets main only
git fetch origin gh-pages:gh-pages   # explicitly fetch docs if needed
```

Deployed automatically via GitHub Pages (Settings → Pages → Source: `gh-pages` / root).
