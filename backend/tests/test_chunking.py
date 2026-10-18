from textwrap import dedent

import pytest

from app.chunking import (
    CLASS_SPLIT_LINES,
    MAX_CHUNK_CHARS,
    MAX_LINES,
    Chunk,
    chunk_file,
    language_of,
)
from app.main import DEMO_DIR


def outline(chunks: list[Chunk]) -> list[tuple[str, str, int, int]]:
    return [(chunk.kind, chunk.name, chunk.start_line, chunk.end_line) for chunk in chunks]


PYTHON = dedent(
    '''\
    """Shopping cart."""

    import os

    MAX_ITEMS = 10


    def add(a, b):
        return a + b


    @cached
    @traced("cart")
    def decorated():
        return 1


    class Small:
        """A small class."""

        def method(self):
            return 2
    '''
)


class TestPython:
    def test_functions_classes_and_module_code_become_chunks(self) -> None:
        chunks = chunk_file("shop", "cart.py", PYTHON)

        assert outline(chunks) == [
            ("module", "<module>", 1, 5),
            ("function", "add", 8, 9),
            ("function", "decorated", 12, 15),
            ("class", "Small", 18, 22),
        ]
        assert chunks[2].content.startswith("@cached\n@traced")
        assert {chunk.language for chunk in chunks} == {"python"}

    def test_a_large_class_is_split_into_its_header_and_methods(self) -> None:
        methods = "\n".join(
            f"    def step_{n}(self):\n        value = {n}\n        return value\n"
            for n in range(12)
        )
        source = f'class Big:\n    """Many steps."""\n\n    LIMIT = 3\n\n{methods}'
        assert len(source.splitlines()) > CLASS_SPLIT_LINES

        chunks = chunk_file("shop", "big.py", source)

        assert outline(chunks)[:3] == [
            ("class", "Big", 1, 4),
            ("method", "Big.step_0", 6, 8),
            ("method", "Big.step_1", 10, 12),
        ]
        assert [chunk.name for chunk in chunks][-1] == "Big.step_11"
        assert len(chunks) == 13

    def test_a_long_function_becomes_overlapping_windows_with_its_name(self) -> None:
        body = "\n".join(f"    x{n} = {n}" for n in range(100))
        chunks = chunk_file("shop", "long.py", f"def long():\n{body}\n")

        assert outline(chunks) == [
            ("function", "long", 1, 60),
            ("function", "long", 51, 101),
        ]
        assert all(len(chunk.content.splitlines()) <= MAX_LINES for chunk in chunks)

    def test_invalid_python_becomes_windows_of_lines(self) -> None:
        source = "def broken(:\n" + "\n".join(f"line {n}" for n in range(50))

        chunks = chunk_file("shop", "broken.py", source)

        assert outline(chunks) == [("block", "lines 1-40", 1, 40), ("block", "lines 33-51", 33, 51)]


JAVASCRIPT = dedent(
    """\
    import { api } from './api';

    /**
     * Formats a price in dollars.
     */
    export function formatPrice(cents) {
      return `$${(cents / 100).toFixed(2)}`;
    }

    export const debounce = (fn, ms) => {
      let timer;
      return (...args) => {
        clearTimeout(timer);
        timer = setTimeout(() => fn(...args), ms);
      };
    };

    class Cart {
      add(item) {
        this.items.push(item);
      }
    }

    export default function App() {
      return null;
    }

    app.listen(3000);
    """
)

TYPESCRIPT = dedent(
    """\
    export interface Book {
      id: string;
      price: number;
    }

    export type Status = "open" | "paid";

    enum Color {
      Red,
      Blue,
    }

    export async function load(id: string): Promise<Book> {
      const response = await fetch(`/books/${id}`);
      return response.json();
    }
    """
)


class TestJavaScriptAndTypeScript:
    def test_functions_arrow_functions_classes_and_module_code(self) -> None:
        chunks = chunk_file("web", "src/app.js", JAVASCRIPT)

        assert outline(chunks) == [
            ("module", "<module>", 1, 1),
            ("function", "formatPrice", 3, 8),
            ("function", "debounce", 10, 16),
            ("class", "Cart", 18, 22),
            ("function", "App", 24, 26),
            ("module", "<module>", 28, 28),
        ]

    def test_the_doc_comment_above_a_definition_is_part_of_it(self) -> None:
        chunk = chunk_file("web", "src/app.js", JAVASCRIPT)[1]

        assert chunk.content.startswith("/**\n * Formats a price in dollars.\n */\nexport function")

    def test_interfaces_types_and_enums_are_type_chunks(self) -> None:
        chunks = chunk_file("web", "src/books.ts", TYPESCRIPT)

        assert outline(chunks) == [
            ("type", "Book", 1, 4),
            ("type", "Status", 6, 6),
            ("type", "Color", 8, 11),
            ("function", "load", 13, 16),
        ]
        assert {chunk.language for chunk in chunks} == {"typescript"}

    def test_tsx_components_are_parsed(self) -> None:
        source = (
            "export function Badge({ n }: { n: number }) {\n"
            "  return <b className='badge'>{n}</b>;\n"
            "}\n"
        )

        chunks = chunk_file("web", "src/Badge.tsx", source)

        assert outline(chunks) == [("function", "Badge", 1, 3)]
        assert chunks[0].language == "tsx"

    def test_a_large_class_is_split_into_its_methods(self) -> None:
        methods = "\n".join(f"  step{n}() {{\n    return {n};\n  }}\n" for n in range(12))
        source = f"export class Steps {{\n  limit = 3;\n\n{methods}}}\n"

        chunks = chunk_file("web", "src/steps.ts", source)

        assert outline(chunks)[:3] == [
            ("class", "Steps", 1, 2),
            ("method", "Steps.step0", 4, 6),
            ("method", "Steps.step1", 8, 10),
        ]
        assert len(chunks) == 13


class TestOtherFiles:
    def test_markdown_becomes_sections_and_ignores_headings_in_code(self) -> None:
        source = dedent(
            """\
            Intro text.

            # Shop

            The shop.

            ## Install

            ```bash
            # not a heading
            pip install shop
            ```

            ## Usage `shop`

            Run it.
            """
        )

        chunks = chunk_file("docs", "README.md", source)

        assert outline(chunks) == [
            ("section", "<intro>", 1, 1),
            ("section", "Shop", 3, 5),
            ("section", "Install", 7, 12),
            ("section", "Usage shop", 14, 16),
        ]

    def test_other_languages_become_overlapping_windows(self) -> None:
        source = "\n".join(f"line {n}" for n in range(1, 101))

        chunks = chunk_file("svc", "main.go", source)

        assert outline(chunks) == [
            ("block", "lines 1-40", 1, 40),
            ("block", "lines 33-72", 33, 72),
            ("block", "lines 65-100", 65, 100),
        ]
        assert chunks[0].language == "go"

    def test_empty_files_have_no_chunks(self) -> None:
        assert chunk_file("shop", "empty.py", "\n  \n") == []

    def test_blank_lines_around_a_unit_are_trimmed(self) -> None:
        chunks = chunk_file("svc", "notes.txt", "\n\nhello\nworld\n\n")

        assert outline(chunks) == [("block", "lines 1-5", 3, 4)]

    def test_very_long_lines_are_cut(self) -> None:
        chunks = chunk_file("svc", "data.json", '{"a": "' + "x" * 20_000 + '"}')

        assert len(chunks[0].content) == MAX_CHUNK_CHARS


class TestChunk:
    def test_key_names_the_code_the_dataset_refers_to(self) -> None:
        chunk = chunk_file("shop", "src/cart.py", PYTHON)[1]

        assert chunk.key == "shop:src/cart.py::add"

    def test_id_is_stable_and_changes_with_the_content(self) -> None:
        first = chunk_file("shop", "cart.py", PYTHON)
        again = chunk_file("shop", "cart.py", PYTHON)
        changed = chunk_file("shop", "cart.py", PYTHON.replace("a + b", "b + a"))

        assert [chunk.id for chunk in first] == [chunk.id for chunk in again]
        assert [a.id == b.id for a, b in zip(first, changed, strict=True)] == [
            True,
            False,
            True,
            True,
        ]

    def test_the_embedded_text_says_what_the_chunk_is(self) -> None:
        chunk = chunk_file("shop", "cart.py", PYTHON)[1]

        assert (
            chunk.embedding_text
            == "python function add in cart.py\ndef add(a, b):\n    return a + b"
        )


@pytest.mark.parametrize(
    ("path", "language"),
    [
        ("src/App.tsx", "tsx"),
        ("lib/util.MJS", "javascript"),
        ("docs/guide.md", "markdown"),
        ("deploy/Dockerfile", "docker"),
        ("README", None),
        (".env", None),
        ("logo.png", None),
    ],
)
def test_language_of(path: str, language: str | None) -> None:
    assert language_of(path) == language


def test_the_demo_codebases_are_chunked_by_definition() -> None:
    root = DEMO_DIR / "bookshop-api"
    cart = chunk_file("bookshop-api", "bookshop/cart.py", (root / "bookshop/cart.py").read_text())
    web = DEMO_DIR / "bookshop-web/src/api/client.ts"
    client = chunk_file("bookshop-web", "src/api/client.ts", web.read_text())

    assert {"Cart", "Cart.add_item", "Cart.apply_coupon", "Cart.shipping"} <= {
        chunk.name for chunk in cart
    }
    assert {"ApiError", "setAccessToken", "refreshAccessToken", "request"} <= {
        chunk.name for chunk in client
    }
