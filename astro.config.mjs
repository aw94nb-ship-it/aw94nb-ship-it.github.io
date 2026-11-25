// @ts-check
import { defineConfig } from 'astro/config';
import sitemap from '@astrojs/sitemap';

// The one place the public URL lives. If the GitHub account is renamed,
// change this line (and the repo name); every link in the site is relative.
const SITE = 'https://aw94nb-ship-it.github.io';

export default defineConfig({
  site: SITE,
  base: '/',
  trailingSlash: 'ignore',
  integrations: [sitemap()],
});
