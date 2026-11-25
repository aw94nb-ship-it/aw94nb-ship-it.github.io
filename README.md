# annie-site

Personal website and essays. Astro static site, deployed to GitHub Pages by GitHub Actions on
push to `main`.

- `npm run dev`: local preview at http://localhost:4321
- `npm run build`: static build into `dist/`
- Essays: `src/content/essays/<slug>.md` with `title`, `description`, `date`, `portfolios`
  (any of `robotics`, `ai-agents`, `applied-ml`, `ab-testing`, `crypto`, `quant-research`).
- Projects: `src/content/projects/<slug>.md` with `title`, `summary`, `date`, `url?`, `portfolios`.
- Portfolio titles and theses: `src/data/portfolios.ts`. Public URL: `site` in `astro.config.mjs`.
