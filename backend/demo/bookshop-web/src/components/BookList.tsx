import { useEffect, useState } from "react";

import { request } from "../api/client";
import { useDebounce } from "../hooks/useDebounce";
import { formatPrice } from "../utils/format";

interface Book {
  id: number;
  title: string;
  author: string;
  price: string;
}

/** Searchable book list: the query is debounced (300 ms) before the API is called. */
export function BookList({ onAdd }: { onAdd: (book: Book) => void }) {
  const [query, setQuery] = useState("");
  const [books, setBooks] = useState<Book[]>([]);
  const debouncedQuery = useDebounce(query, 300);

  useEffect(() => {
    let active = true;
    request<{ items: Book[] }>(`/books?q=${encodeURIComponent(debouncedQuery)}`).then((page) => {
      if (active) setBooks(page.items);
    });
    return () => {
      active = false;
    };
  }, [debouncedQuery]);

  return (
    <section>
      <input
        value={query}
        onChange={(event) => setQuery(event.target.value)}
        placeholder="Search books"
      />
      <ul>
        {books.map((book) => (
          <li key={book.id}>
            {book.title} by {book.author}: {formatPrice(book.price)}
            <button onClick={() => onAdd(book)}>Add to cart</button>
          </li>
        ))}
      </ul>
    </section>
  );
}
