// The six learning portfolios. Essays and projects join a portfolio by listing its slug
// in their `portfolios` frontmatter, so these pages fill themselves.
export const PORTFOLIOS = [
  {
    slug: 'robotics',
    title: 'Robotics',
    thesis:
      'Robotics is held back by data, not algorithms. I care about the unglamorous parts: sensing, evaluation, and the gap between a demo and a robot that works on an ordinary Tuesday.',
  },
  {
    slug: 'ai-agents',
    title: 'AI agents',
    thesis:
      'An agent is a loop, not a model. What it can see, what it is allowed to change, and how fast it finds out it was wrong decide whether it is useful.',
  },
  {
    slug: 'applied-ml',
    title: 'Applied ML',
    thesis:
      'In production, the model is the easy part. The work is the data, the evaluation, and the discipline to ship the simplest thing that holds up.',
  },
  {
    slug: 'ab-testing',
    title: 'A/B testing and product',
    thesis:
      'Experiments keep a product team honest. The craft is choosing what to measure, when to stop, and when a significant result still isn\'t worth acting on.',
  },
  {
    slug: 'crypto',
    title: 'Crypto',
    thesis:
      'Open ledgers make crypto the most transparent market there is. Every trade and incentive is visible, which makes it great to study and easy to fool yourself in.',
  },
  {
    slug: 'quant-research',
    title: 'Quantitative research',
    thesis:
      'Edge is small, noisy and decays. Research is the discipline of not believing your own backtest until the evidence forces you to.',
  },
] as const;

export type PortfolioSlug = (typeof PORTFOLIOS)[number]['slug'];
export const PORTFOLIO_SLUGS = PORTFOLIOS.map((p) => p.slug) as [PortfolioSlug, ...PortfolioSlug[]];

export function portfolioBySlug(slug: string) {
  return PORTFOLIOS.find((p) => p.slug === slug);
}
