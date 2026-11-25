// The six learning portfolios. Essays and projects join a portfolio by listing its slug
// in their `portfolios` frontmatter, so these pages fill themselves.
//
// DRAFT COPY: the theses below are placeholders written by an agent for Annie to replace
// in her own words before anything is published.
export const PORTFOLIOS = [
  {
    slug: 'robotics',
    title: 'Robotics',
    thesis:
      'Robots are a measurement problem before they are a control problem. Most progress comes from better data about the physical world, not cleverer policies.',
  },
  {
    slug: 'ai-agents',
    title: 'AI agents',
    thesis:
      'An agent is only as good as the loop around it: what it can observe, what it is allowed to change, and how quickly it finds out it was wrong.',
  },
  {
    slug: 'applied-ml',
    title: 'Applied ML',
    thesis:
      'In production, the model is the easy part. The work is in the data, the evaluation, and the discipline to ship the simplest thing that holds up.',
  },
  {
    slug: 'ab-testing',
    title: 'A/B testing and product',
    thesis:
      'Experiments are how a product team stays honest. Good ones change decisions; most of the craft is choosing what to measure and when to stop.',
  },
  {
    slug: 'crypto',
    title: 'Crypto',
    thesis:
      'Markets with open ledgers are a rare laboratory: every trade and incentive is visible, which makes them unusually good to study and unusually easy to fool yourself in.',
  },
  {
    slug: 'quant-research',
    title: 'Quantitative research',
    thesis:
      'Edge is small, noisy and decays. Research is the process of not believing your own backtest until the evidence forces you to.',
  },
] as const;

export type PortfolioSlug = (typeof PORTFOLIOS)[number]['slug'];
export const PORTFOLIO_SLUGS = PORTFOLIOS.map((p) => p.slug) as [PortfolioSlug, ...PortfolioSlug[]];

export function portfolioBySlug(slug: string) {
  return PORTFOLIOS.find((p) => p.slug === slug);
}
