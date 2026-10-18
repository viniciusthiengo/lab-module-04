// Answers cite their sources as [1], [2]... These become links to the source cards, except
// inside code, where brackets are code (`items[1]`).

const CODE = /(```[\s\S]*?```|~~~[\s\S]*?~~~|`[^`\n]*`)/g;
const CITATION = /\[(\d+(?:\s*,\s*\d+)*)\](?!\()/g;

/** The prefix of the citation links: `#source-3` points to the third source. */
export const CITATION_PREFIX = "#source-";

/**
 * `markdown` with its citations of sources 1 to `sources` as links (`[1](#source-1)`); other
 * numbers and everything inside code stay as they are. `[1, 2]` becomes two links.
 */
export function linkCitations(markdown: string, sources: number): string {
  return markdown
    .split(CODE)
    .map((part, index) => (index % 2 === 1 ? part : linkPart(part, sources)))
    .join("");
}

function linkPart(text: string, sources: number): string {
  return text.replace(CITATION, (whole, list: string) => {
    const numbers = list.split(",").map((value) => Number(value.trim()));
    if (!numbers.every((n) => Number.isInteger(n) && n >= 1 && n <= sources)) return whole;
    return numbers.map((n) => `[${n}](${CITATION_PREFIX}${n})`).join(" ");
  });
}

/** The source number of a citation link, or `null` for any other link. */
export function citedSource(href: string | undefined): number | null {
  if (!href?.startsWith(CITATION_PREFIX)) return null;
  const number = Number(href.slice(CITATION_PREFIX.length));
  return Number.isInteger(number) && number >= 1 ? number : null;
}
