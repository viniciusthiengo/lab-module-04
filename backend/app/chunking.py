"""Code-aware chunking: a file becomes chunks that follow its structure.

Python is parsed with its own ``ast`` (functions, classes, and the methods of large classes),
JavaScript and TypeScript with tree-sitter (functions, arrow functions, classes, methods,
interfaces and types, with the doc comment above them), Markdown by sections; other files
become overlapping windows of lines. Code outside definitions (imports, constants, route
registrations) is kept as ``module`` chunks. Each chunk keeps its line range and a name that
identifies it in its file, such as ``Cart.add_item``.
"""

import ast
import hashlib
import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import NamedTuple

import tree_sitter_javascript
import tree_sitter_typescript
from tree_sitter import Language, Node, Parser

MAX_LINES = 80  # a definition longer than this is split into windows
CLASS_SPLIT_LINES = 40  # a class longer than this becomes one chunk per method
WINDOW_LINES = 60
WINDOW_OVERLAP = 10
GENERIC_WINDOW_LINES = 40
GENERIC_WINDOW_OVERLAP = 8
MAX_CHUNK_CHARS = 8_000

LANGUAGES = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".ts": "typescript",
    ".mts": "typescript",
    ".tsx": "tsx",
    ".md": "markdown",
    ".markdown": "markdown",
    ".go": "go",
    ".java": "java",
    ".kt": "kotlin",
    ".rb": "ruby",
    ".php": "php",
    ".rs": "rust",
    ".c": "c",
    ".h": "c",
    ".cc": "cpp",
    ".cpp": "cpp",
    ".hpp": "cpp",
    ".cs": "csharp",
    ".swift": "swift",
    ".scala": "scala",
    ".sql": "sql",
    ".sh": "shell",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
    ".html": "html",
    ".css": "css",
    ".txt": "text",
    ".rst": "text",
    ".ini": "text",
    ".cfg": "text",
}
FILE_NAMES = {"Dockerfile": "docker", "Makefile": "make"}

_TREE_SITTER = {
    "javascript": Language(tree_sitter_javascript.language()),
    "typescript": Language(tree_sitter_typescript.language_typescript()),
    "tsx": Language(tree_sitter_typescript.language_tsx()),
}


def language_of(path: str) -> str | None:
    """The language of a file, or ``None`` when it is not a supported text file."""
    name = path.rsplit("/", 1)[-1]
    if name in FILE_NAMES:
        return FILE_NAMES[name]
    dot = name.rfind(".")
    return LANGUAGES.get(name[dot:].lower()) if dot > 0 else None


@dataclass(frozen=True)
class Chunk:
    """A piece of a file, with where it comes from.

    ``kind`` is function, class, method, type, module, section or block.
    """

    codebase: str
    path: str
    language: str
    kind: str
    name: str
    start_line: int
    end_line: int
    content: str

    @property
    def key(self) -> str:
        """``codebase:path::name``: what the evaluation dataset refers to."""
        return f"{self.codebase}:{self.path}::{self.name}"

    @property
    def id(self) -> str:
        """Stable while the chunk does not change, so re-indexing skips unchanged chunks."""
        digest = hashlib.sha1(
            f"{self.codebase}\0{self.path}\0{self.start_line}\0{self.end_line}\0{self.content}".encode()
        )
        return digest.hexdigest()[:20]

    @property
    def embedding_text(self) -> str:
        """What is embedded: a header that says what the chunk is, then the code."""
        return f"{self.language} {self.kind} {self.name} in {self.path}\n{self.content}"


class _Unit(NamedTuple):
    kind: str
    name: str
    start: int  # 1-based, inclusive
    end: int


def chunk_file(codebase: str, path: str, content: str) -> list[Chunk]:
    """The chunks of one file (none for an empty file)."""
    language = language_of(path) or "text"
    lines = content.splitlines()
    if not any(line.strip() for line in lines):
        return []
    units: list[_Unit] | None = None
    if language == "python":
        units = _python_units(content)
    elif language in _TREE_SITTER:
        units = _tree_sitter_units(content, language)
    elif language == "markdown":
        units = _markdown_units(lines)
    if units is None:
        units = _windows("block", None, 1, len(lines), GENERIC_WINDOW_LINES, GENERIC_WINDOW_OVERLAP)
    chunks = []
    for unit in sorted(units, key=lambda unit: (unit.start, unit.end)):
        start, end = _trim(lines, unit.start, unit.end)
        if start > end:
            continue
        text = "\n".join(lines[start - 1 : end])[:MAX_CHUNK_CHARS]
        chunks.append(Chunk(codebase, path, language, unit.kind, unit.name, start, end, text))
    return chunks


def _trim(lines: list[str], start: int, end: int) -> tuple[int, int]:
    """The range without its leading and trailing blank lines."""
    while start <= end and not lines[start - 1].strip():
        start += 1
    while end >= start and not lines[end - 1].strip():
        end -= 1
    return start, end


def _sized(kind: str, name: str, start: int, end: int) -> list[_Unit]:
    """One unit, or overlapping windows with the same name when it is too long."""
    if end - start + 1 <= MAX_LINES:
        return [_Unit(kind, name, start, end)]
    return _windows(kind, name, start, end, WINDOW_LINES, WINDOW_OVERLAP)


def _windows(
    kind: str, name: str | None, start: int, end: int, size: int, overlap: int
) -> list[_Unit]:
    units = []
    first = start
    while first <= end:
        last = min(first + size - 1, end)
        units.append(_Unit(kind, name or f"lines {first}-{last}", first, last))
        if last == end:
            break
        first = last - overlap + 1
    return units


def _module_units(line_numbers: Iterable[int]) -> list[_Unit]:
    """Module-level code, in runs of consecutive lines (blank lines may separate them)."""
    units: list[_Unit] = []
    run: list[int] = []
    for number in sorted(set(line_numbers)):
        if run and number - run[-1] > 3:
            units += _sized("module", "<module>", run[0], run[-1])
            run = []
        run.append(number)
    if run:
        units += _sized("module", "<module>", run[0], run[-1])
    return units


# --- Python -------------------------------------------------------------------------------


def _python_units(source: str) -> list[_Unit] | None:
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return None  # not valid Python 3: windows of lines instead
    units: list[_Unit] = []
    module_lines: list[int] = []
    for node in tree.body:
        start, end = _span(node)
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            units += _sized("function", node.name, start, end)
        elif isinstance(node, ast.ClassDef):
            units += _python_class(node, start, end)
        else:
            module_lines += range(start, end + 1)
    return units + _module_units(module_lines)


def _span(node: ast.stmt) -> tuple[int, int]:
    """The lines of a statement, decorators included."""
    decorators = getattr(node, "decorator_list", [])
    start = min([node.lineno, *(decorator.lineno for decorator in decorators)])
    return start, node.end_lineno or node.lineno


def _python_class(node: ast.ClassDef, start: int, end: int) -> list[_Unit]:
    if end - start + 1 <= CLASS_SPLIT_LINES:
        return [_Unit("class", node.name, start, end)]
    methods = [n for n in node.body if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef)]
    if not methods:
        return _sized("class", node.name, start, end)
    # The class chunk keeps the signature, docstring and attributes; each method follows.
    header_end = min(_span(method)[0] for method in methods) - 1
    units = [_Unit("class", node.name, start, header_end)]
    for method in methods:
        units += _sized("method", f"{node.name}.{method.name}", *_span(method))
    return units


# --- JavaScript and TypeScript ------------------------------------------------------------

_FUNCTIONS = {"function_declaration", "generator_function_declaration"}
_CLASSES = {"class_declaration", "abstract_class_declaration"}
_TYPES = {"interface_declaration", "type_alias_declaration", "enum_declaration"}
_VARIABLES = {"lexical_declaration", "variable_declaration"}
_FUNCTION_VALUES = {"arrow_function", "function_expression", "function", "call_expression"}


def _tree_sitter_units(source: str, language: str) -> list[_Unit]:
    root = Parser(_TREE_SITTER[language]).parse(source.encode("utf-8")).root_node
    units: list[_Unit] = []
    module_lines: list[int] = []
    comment_start: int | None = None
    for node in root.named_children:
        start, end = node.start_point.row + 1, node.end_point.row + 1
        if node.type == "comment":
            comment_start = comment_start or start
            module_lines += range(start, end + 1)
            continue
        found = _js_units(node)
        if found:
            if comment_start is not None and comment_start < start:
                # The doc comment right above a definition belongs to it.
                module_lines = [line for line in module_lines if line < comment_start]
                first = found[0]
                found[0] = first._replace(start=min(first.start, comment_start))
            units += found
        else:
            module_lines += range(start, end + 1)
        comment_start = None
    return units + _module_units(module_lines)


def _js_units(node: Node) -> list[_Unit]:
    """The units of a top-level statement; empty when it is not a definition."""
    start, end = node.start_point.row + 1, node.end_point.row + 1
    declaration = node
    if node.type == "export_statement":
        inner = node.child_by_field_name("declaration")
        if inner is None:
            inner = next((c for c in node.named_children if c.type in _FUNCTIONS | _CLASSES), None)
        if inner is None:
            return []
        declaration = inner
    kind = declaration.type
    if kind in _FUNCTIONS:
        return _sized("function", _name(declaration), start, end)
    if kind in _TYPES:
        return _sized("type", _name(declaration), start, end)
    if kind in _CLASSES:
        return _js_class(declaration, start, end)
    if kind in _VARIABLES:
        declarators = [c for c in declaration.named_children if c.type == "variable_declarator"]
        value = declarators[0].child_by_field_name("value") if declarators else None
        if value is not None and value.type in _FUNCTION_VALUES:
            return _sized("function", _name(declarators[0]), start, end)
    return []


def _js_class(node: Node, start: int, end: int) -> list[_Unit]:
    name = _name(node)
    body = node.child_by_field_name("body")
    methods = [c for c in body.named_children if c.type == "method_definition"] if body else []
    if end - start + 1 <= CLASS_SPLIT_LINES or not methods:
        return _sized("class", name, start, end)
    units = [_Unit("class", name, start, methods[0].start_point.row)]
    for method in methods:
        units += _sized(
            "method",
            f"{name}.{_name(method)}",
            method.start_point.row + 1,
            method.end_point.row + 1,
        )
    return units


def _name(node: Node) -> str:
    name = node.child_by_field_name("name")
    return name.text.decode("utf-8") if name is not None and name.text else "<anonymous>"


# --- Markdown ------------------------------------------------------------------------------

_HEADING = re.compile(r"^(#{1,3})\s+(.+?)\s*#*\s*$")


def _markdown_units(lines: list[str]) -> list[_Unit]:
    """One unit per section (headings of level 1 to 3), text before the first as intro."""
    headings: list[tuple[int, str]] = []
    in_code = False
    for number, line in enumerate(lines, start=1):
        if line.lstrip().startswith(("```", "~~~")):
            in_code = not in_code
            continue
        match = None if in_code else _HEADING.match(line)
        if match:
            headings.append((number, match.group(2).replace("`", "")))
    units: list[_Unit] = []
    first = headings[0][0] if headings else len(lines) + 1
    if first > 1:
        units += _sized("section", "<intro>", 1, first - 1)
    for index, (start, title) in enumerate(headings):
        end = headings[index + 1][0] - 1 if index + 1 < len(headings) else len(lines)
        units += _sized("section", title, start, end)
    return units
