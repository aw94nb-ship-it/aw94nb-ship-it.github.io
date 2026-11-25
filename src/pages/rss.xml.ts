import rss from '@astrojs/rss';
import type { APIContext } from 'astro';
import { SITE_TITLE, SITE_DESCRIPTION } from '../data/site';
import { getEssays, url } from '../lib/content';

export async function GET(context: APIContext) {
  const essays = await getEssays();
  return rss({
    title: SITE_TITLE,
    description: SITE_DESCRIPTION,
    site: context.site!,
    items: essays.map((e) => ({
      title: e.data.title,
      description: e.data.description,
      pubDate: e.data.date,
      link: url(`essays/${e.id}/`),
    })),
  });
}
