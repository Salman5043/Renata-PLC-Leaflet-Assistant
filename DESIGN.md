# DESIGN — Renata PLC Leaflet Assistant v4

## Architecture

Five Renata PDFs 
  ↓ PyMuPDF
Section-aware parsing + product/ingredient/page metadata
  ↓ Recursive chunking
Hugging Face all-MiniLM-L6-v2
  ↓
Local ChromaDB

Question → FastAPI → LangGraph
                    ↓
             Question router
             ↙            ↘
        Greeting          RAG question
           ↓                   ↓
       Groq LLM        Product detection
                              ↓
                    Section intent detection
                              ↓
                     Product-aware retrieval
                              ↓
                 Dense + lexical + section ranking
                              ↓
                      Evidence diversification
                              ↓
                       Context builder
                              ↓
                       Groq generation
                              ↓
                    Citation sanity check
                              ↓
                    Groq grounding check
                       ↙              ↘
                    PASS              FAIL
                     ↓                  ↓
              cited answer          abstain

## Chunking strategy

The leaflet is semantically structured, so I preserve numbered section boundaries before recursive character chunking. This keeps concepts such as a product's indication, dosage, precautions and side effects together as much as possible. Default chunk size is 1000 characters with 150-character overlap. These are heuristics for this small corpus, not universal constants.

## Retrieval

When a product is explicitly named, Chroma metadata filtering restricts retrieval to that product. This is important because the five leaflets reuse headings such as “How to take” and “Possible side effects”. Candidates are ranked using dense similarity plus lexical overlap, with a small section-match bonus. I intentionally avoid a universal similarity threshold because vector scores are not calibrated confidence values.

## Grounding and “not in the documents”

The generator receives only retrieved leaflet chunks and must cite factual statements with `[C1]`, `[C2]`, etc. Citation labels are validated deterministically. A second Groq call then checks whether every factual claim is actually supported by the supplied context. If citation validation or grounding validation fails — including a validator API failure — the graph fails closed and returns: `I don't have that information in the provided documents.`

This is stronger than relying on a prompt alone: retrieval controls the evidence, generation is constrained to it, and validation creates an explicit abstention gate.

## Model choices and trade-offs

- **Groq:** fast hosted inference and acceptable for the assignment's free-tier option. Trade-off: network/API dependency.
- **Hugging Face `all-MiniLM-L6-v2`:** small local CPU-friendly embedding model, appropriate for five leaflets. Trade-off: less powerful than larger embedding models.
- **ChromaDB:** local and simple. Trade-off: not intended here as a production distributed vector service.
- **LangChain:** handles document, embedding, vector-store and LLM abstractions.
- **LangGraph:** makes retrieval, generation, validation and abstention explicit stateful nodes instead of hiding the workflow in one chain.

## What I would improve

With more time: BM25 + dense retrieval with a cross-encoder reranker; table-aware PDF extraction; automated retrieval/faithfulness metrics; incremental indexing; tracing; and deterministic claim-to-citation verification.

## Key principle

The Renata leaflets are the source of truth. The LLM is only a synthesis layer.
