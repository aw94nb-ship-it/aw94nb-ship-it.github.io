import { defineCollection } from 'astro:content';
import { glob } from 'astro/loaders';
import { z } from 'astro/zod';
import { PORTFOLIO_SLUGS } from './data/portfolios';

const essays = defineCollection({
  loader: glob({ base: './src/content/essays', pattern: '**/*.{md,mdx}' }),
  schema: z.object({
    title: z.string(),
    description: z.string(),
    date: z.coerce.date(),
    updated: z.coerce.date().optional(),
    portfolios: z.array(z.enum(PORTFOLIO_SLUGS)).default([]),
    // Drafts never render. Unapproved essays should not be in this repo at all.
    draft: z.boolean().default(false),
  }),
});

// Independent personal work, shown on portfolio pages.
const projects = defineCollection({
  loader: glob({ base: './src/content/projects', pattern: '**/*.{md,mdx}' }),
  schema: z.object({
    title: z.string(),
    summary: z.string(),
    date: z.coerce.date(),
    url: z.string().url().optional(),
    portfolios: z.array(z.enum(PORTFOLIO_SLUGS)).default([]),
    draft: z.boolean().default(false),
  }),
});

export const collections = { essays, projects };
