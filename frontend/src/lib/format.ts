// How numbers, costs, durations and code locations are shown.

/** `0.8433` → "84%". */
export function percent(value: number): string {
  return `${Math.round(value * 100)}%`;
}

/** A metric between 0 and 1: `0.7778` → "0.778". */
export function score(value: number): string {
  return value.toFixed(3);
}

/** A judge's mean grade: `4.6667` → "4.67". */
export function grade(value: number): string {
  return value.toFixed(2);
}

/** US dollars, with the precision small amounts need: "$0.0042", "$0.35", "$1.20". */
export function usd(value: number): string {
  if (value === 0) return "$0";
  if (value < 0.01) return `$${value.toFixed(4)}`;
  if (value < 1) return `$${value.toFixed(3)}`;
  return `$${value.toFixed(2)}`;
}

/** `850` → "850 ms", `12_300` → "12.3 s", `95_000` → "1 min 35 s". */
export function duration(ms: number): string {
  if (ms < 1000) return `${Math.round(ms)} ms`;
  if (ms < 60_000) return `${(ms / 1000).toFixed(1)} s`;
  const seconds = Math.round(ms / 1000);
  return `${Math.floor(seconds / 60)} min ${seconds % 60} s`;
}

/** The lines of a chunk: "L18–22", or "L7" for one line. */
export function lines(start: number, end: number): string {
  return start === end ? `L${start}` : `L${start}–${end}`;
}

/** `1234` → "1,234". */
export function count(value: number): string {
  return value.toLocaleString("en-US");
}

/** "1 file", "3 files". */
export function plural(value: number, word: string, many = `${word}s`): string {
  return `${count(value)} ${value === 1 ? word : many}`;
}

/** A Unix time (seconds) as "Oct 17, 2026, 11:40 PM" in the visitor's time zone. */
export function dateTime(seconds: number): string {
  return new Date(seconds * 1000).toLocaleString("en-US", {
    dateStyle: "medium",
    timeStyle: "short",
  });
}

/** A chunk key `codebase:path::name` split into its parts. */
export function splitKey(key: string): { codebase: string; path: string; name: string } {
  const [location, name = ""] = key.split("::");
  const colon = location.indexOf(":");
  return { codebase: location.slice(0, colon), path: location.slice(colon + 1), name };
}
