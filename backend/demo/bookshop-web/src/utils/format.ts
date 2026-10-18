const priceFormatter = new Intl.NumberFormat("en-US", { style: "currency", currency: "USD" });

/** "$1,234.50": the API sends prices as strings with two decimals. */
export function formatPrice(value: string | number): string {
  return priceFormatter.format(typeof value === "string" ? Number.parseFloat(value) : value);
}

/** "Mar 5, 2026" in the visitor's locale. */
export function formatDate(iso: string, locale?: string): string {
  return new Date(iso).toLocaleDateString(locale, { year: "numeric", month: "short", day: "numeric" });
}
