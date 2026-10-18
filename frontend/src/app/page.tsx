import { RagApp } from "@/components/RagApp";

export default function Home() {
  return (
    <main className="page">
      <header className="page-header">
        <h1>Codebase RAG</h1>
        <p>
          Ask questions about your code. Files are split into functions, classes and sections,
          embedded into ChromaDB and found with vector and keyword search; Claude reranks them and
          answers with citations. The evaluation measures every retrieval setting on a dataset of
          questions.
        </p>
      </header>
      <RagApp />
      <footer className="page-footer">
        Embeddings: all-MiniLM-L6-v2 (sentence-transformers, ONNX) · Vector store: ChromaDB ·
        Answers, reranking and judging: Claude
      </footer>
    </main>
  );
}
