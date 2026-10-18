import { useCallback, useState } from "react";

import { request } from "../api/client";

export interface CartLine {
  bookId: number;
  title: string;
  quantity: number;
}

/**
 * Cart state with optimistic updates: a book appears in the cart at once, and the change is
 * rolled back if the API refuses it.
 */
export function useCart(initial: CartLine[] = []) {
  const [lines, setLines] = useState<CartLine[]>(initial);
  const [error, setError] = useState<string | null>(null);

  const addBook = useCallback(
    async (bookId: number, title: string) => {
      const previous = lines;
      setLines((current) => {
        const existing = current.find((line) => line.bookId === bookId);
        return existing
          ? current.map((line) =>
              line.bookId === bookId ? { ...line, quantity: line.quantity + 1 } : line,
            )
          : [...current, { bookId, title, quantity: 1 }];
      });
      try {
        await request("/cart/items", {
          method: "POST",
          body: JSON.stringify({ book_id: bookId, quantity: 1 }),
        });
        setError(null);
      } catch (caught) {
        setLines(previous);
        setError(caught instanceof Error ? caught.message : "Could not add the book");
      }
    },
    [lines],
  );

  const itemCount = lines.reduce((sum, line) => sum + line.quantity, 0);
  return { lines, itemCount, error, addBook };
}
