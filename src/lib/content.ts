import { getCollection } from 'astro:content';

const byDateDesc = (a: { data: { date: Date } }, b: { data: { date: Date } }) =>
  b.data.date.valueOf() - a.data.date.valueOf();

export async function getEssays() {
  return (await getCollection('essays', ({ data }) => !data.draft)).sort(byDateDesc);
}

export async function getProjects() {
  return (await getCollection('projects', ({ data }) => !data.draft)).sort(byDateDesc);
}

// Join a path onto the configured base so links survive a base change.
export function url(path: string) {
  const base = import.meta.env.BASE_URL.replace(/\/$/, '');
  return `${base}/${path.replace(/^\//, '')}`;
}

export function formatDate(d: Date) {
  return d.toLocaleDateString('en-US', { year: 'numeric', month: 'short', day: 'numeric', timeZone: 'UTC' });
}
