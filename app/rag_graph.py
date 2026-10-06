from __future__ import annotations

import re
import time
from typing import Any, TypedDict

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph

from .config import (
    ABSTAIN,
    CANDIDATE_K,
    DENSE_WEIGHT,
    LEXICAL_WEIGHT,
    MAX_CONTEXT_CHARS,
    MAX_EVIDENCE_DOCUMENTS,
    GROQ_MODEL,
    SECTION_WEIGHT,
    TOP_K,
)
from .embeddings import get_vectorstore


# ============================================================
# PRODUCT ALIASES
# ============================================================

PRODUCT_ALIASES = {
    "doxicap": "Doxicap 100 mg",
    "doxicap 100": "Doxicap 100 mg",
    "doxicap 100 mg": "Doxicap 100 mg",

    "fenadin": "Fenadin 120 mg",
    "fenadin 120": "Fenadin 120 mg",
    "fenadin 120 mg": "Fenadin 120 mg",

    "maxpro": "Maxpro 20 mg",
    "maxpro 20": "Maxpro 20 mg",
    "maxpro 20 mg": "Maxpro 20 mg",

    "rolac": "Rolac 10 mg",
    "rolac 10": "Rolac 10 mg",
    "rolac 10 mg": "Rolac 10 mg",

    "rolip": "Rolip 10 mg",
    "rolip 10": "Rolip 10 mg",
    "rolip 10 mg": "Rolip 10 mg",
}


# ============================================================
# SECTION DETECTION
# ============================================================

SECTION_PATTERNS = {
    "what": [
        "what is",
        "what are",
        "used for",
        "use of",
        "indication",
        "indications",
        "purpose",
    ],

    "before": [
        "before taking",
        "before you take",
        "precaution",
        "precautions",
        "warning",
        "warnings",
        "contraindication",
        "contraindications",
    ],

    "how": [
        "how to take",
        "how should i take",
        "dose",
        "dosage",
        "daily dose",
        "adult dose",
        "children dose",
        "maximum dose",
    ],

    "side_effects": [
        "side effect",
        "side effects",
        "adverse effect",
        "adverse effects",
    ],

    "pregnancy": [
        "pregnancy",
        "pregnant",
        "breast feeding",
        "breastfeeding",
        "breast-feed",
        "breast-fed",
        "nursing",
        "lactation",
    ],

    "storage": [
        "store",
        "storage",
        "keep",
    ],
}


SECTION_NAME_MATCHES = {
    "what": [
        "what is",
        "used for",
    ],

    "before": [
        "before",
    ],

    "how": [
        "how to take",
        "dosage",
        "dose",
    ],

    "side_effects": [
        "side effects",
        "adverse effects",
    ],

    "pregnancy": [
        "pregnancy",
        "breast-feeding",
        "breast feeding",
        "breastfeeding",
        "nursing",
        "lactation",
    ],

    "storage": [
        "store",
        "storage"
    ],
}


# ============================================================
# GREETING DETECTION
# ============================================================

GREETING_PATTERNS = {
    "hi",
    "hello",
    "hey",
    "hi there",
    "hello there",
    "hey there",
    "good morning",
    "good afternoon",
    "good evening",
    "good day",
}


# ============================================================
# GRAPH STATE
# ============================================================

class GraphState(TypedDict, total=False):
    question: str
    product_filter: str | None
    section_intent: str | None
    documents: list[dict[str, Any]]
    answer: str
    citations: list[dict[str, Any]]
    grounded: bool
    citation_valid: bool
    abstained: bool
    metrics: dict[str, Any]


# ============================================================
# TEXT UTILITIES
# ============================================================

def normalize_text(text: str) -> str:
    text = (text or "").lower()
    text = re.sub(r"[^a-z0-9\s-]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> set[str]:
    return {
        token
        for token in normalize_text(text).split()
        if len(token) > 1
    }


def lexical_overlap(question: str, content: str) -> float:
    q = tokenize(question)
    c = tokenize(content)

    if not q or not c:
        return 0.0

    return len(q & c) / len(q)


# ============================================================
# GREETING
# ============================================================

def is_greeting(question: str) -> bool:
    q = normalize_text(question)

    if not q:
        return False

    if q in GREETING_PATTERNS:
        return True

    words = q.split()

    if len(words) <= 3 and words[0] in {
        "hi",
        "hello",
        "hey",
    }:
        return True

    return False


# ============================================================
# PRODUCT DETECTION
# ============================================================

def detect_product_filter(question: str) -> str | None:
    q = normalize_text(question)

    for alias, product in sorted(
        PRODUCT_ALIASES.items(),
        key=lambda x: len(x[0]),
        reverse=True,
    ):
        if re.search(
            rf"\b{re.escape(alias)}\b",
            q,
        ):
            return product

    return None


def detect_product(question: str) -> str | None:
    return detect_product_filter(question)


# ============================================================
# SECTION DETECTION
# ============================================================

def detect_section_intent(question: str) -> str | None:
    q = normalize_text(question)

    for section, patterns in SECTION_PATTERNS.items():
        for pattern in patterns:
            if pattern in q:
                return section

    return None


def section_match_score(
    document: dict[str, Any],
    intent: str | None,
) -> float:

    if not intent:
        return 0.0

    section = normalize_text(
        str(document.get("section", ""))
    )

    matches = SECTION_NAME_MATCHES.get(
        intent,
        [],
    )

    for match in matches:
        if match in section:
            return 1.0

    return 0.0


# ============================================================
# VECTOR SCORE
# ============================================================

def distance_to_similarity(
    distance: float,
) -> float:

    return max(
        0.0,
        min(
            1.0,
            1.0 / (
                1.0 +
                max(
                    0.0,
                    float(distance),
                )
            ),
        ),
    )


# ============================================================
# DOCUMENT CONVERSION
# ============================================================

def document_to_dict(
    document: Document,
    dense: float,
    lexical: float,
    section: float,
) -> dict[str, Any]:

    metadata = document.metadata or {}

    score = (
        DENSE_WEIGHT * dense
        + LEXICAL_WEIGHT * lexical
        + SECTION_WEIGHT * section
    )

    return {
        "content": document.page_content.strip(),

        "source": metadata.get(
            "source",
            "unknown",
        ),

        "product": metadata.get(
            "product",
            "unknown",
        ),

        "active_ingredient": metadata.get(
            "active_ingredient",
            "unknown",
        ),

        "page": metadata.get("page"),

        "section": metadata.get(
            "section",
            "unknown",
        ),

        "chunk_id": metadata.get(
            "chunk_id"
        ),

        "dense_score": round(
            dense,
            4,
        ),

        "lexical_score": round(
            lexical,
            4,
        ),

        "section_score": round(
            section,
            4,
        ),

        "score": round(
            score,
            4,
        ),
    }


# ============================================================
# SECTION-MATCHING DOCUMENTS
# ============================================================

def get_section_documents(
    candidates: list[dict[str, Any]],
    intent: str | None,
) -> list[dict[str, Any]]:

    if not intent:
        return []

    matched = [
        item
        for item in candidates
        if section_match_score(
            item,
            intent,
        ) > 0
    ]

    # Highest score first
    matched.sort(
        key=lambda x: x["score"],
        reverse=True,
    )

    return matched


# ============================================================
# BEST CHUNK PER PRODUCT
# ============================================================

def best_per_product(
    candidates: list[dict[str, Any]],
    intent: str | None,
) -> list[dict[str, Any]]:

    if not intent:
        return []

    grouped: dict[
        str,
        list[dict[str, Any]]
    ] = {}

    for item in candidates:

        if section_match_score(
            item,
            intent,
        ) <= 0:
            continue

        product = str(
            item.get(
                "product",
                "unknown",
            )
        )

        grouped.setdefault(
            product,
            [],
        ).append(item)

    best = []

    for product, items in grouped.items():

        items.sort(
            key=lambda x: x["score"],
            reverse=True,
        )

        best.append(
            items[0]
        )

    return sorted(
        best,
        key=lambda x: x["score"],
        reverse=True,
    )


# ============================================================
# RETRIEVAL NODE
# ============================================================

def retrieve_node(
    state: GraphState,
) -> GraphState:

    started = time.perf_counter()

    question = state["question"]

    product = detect_product_filter(
        question
    )

    intent = detect_section_intent(
        question
    )

    store = get_vectorstore()

    # --------------------------------------------------------
    # VECTOR RETRIEVAL
    # --------------------------------------------------------

    kwargs: dict[str, Any] = {
        "k": max(
            CANDIDATE_K,
            TOP_K,
        )
    }

    if product:
        kwargs["filter"] = {
            "product": product
        }

    results = (
        store
        .similarity_search_with_score(
            question,
            **kwargs,
        )
    )

    candidates = []

    for doc, distance in results:

        dense = distance_to_similarity(
            distance
        )

        lexical = lexical_overlap(
            question,
            doc.page_content,
        )

        section = section_match_score(
            {
                "section": (
                    doc.metadata or {}
                ).get(
                    "section",
                    "",
                )
            },
            intent,
        )

        candidates.append(
            document_to_dict(
                doc,
                dense,
                lexical,
                section,
            )
        )

    candidates.sort(
        key=lambda x: x["score"],
        reverse=True,
    )

    # --------------------------------------------------------
    # PRODUCT FILTER
    # --------------------------------------------------------

    if product:

        candidates = [
            item
            for item in candidates
            if item.get("product")
            == product
        ]

    # ========================================================
    # IMPORTANT FIX
    #
    # GENERAL SECTION QUESTION
    #
    # Example:
    #
    # "Use in pregnancy and breast-feeding"
    #
    # If there are 5 products and each has a pregnancy
    # section, we MUST preserve all 5.
    #
    # We do NOT collapse them into one document.
    # ========================================================

    if intent and not product:

        section_documents = (
            get_section_documents(
                candidates,
                intent,
            )
        )

        if section_documents:

            # Remove duplicate chunks while preserving
            # the best chunk for each product/section.
            unique: dict[
                tuple[str, str],
                dict[str, Any]
            ] = {}

            for item in section_documents:

                key = (
                    str(
                        item.get(
                            "product",
                            "unknown",
                        )
                    ),
                    str(
                        item.get(
                            "section",
                            "unknown",
                        )
                    ),
                )

                existing = unique.get(key)

                if (
                    existing is None
                    or item["score"]
                    > existing["score"]
                ):
                    unique[key] = item

            evidence = list(
                unique.values()
            )

            evidence.sort(
                key=lambda x: x["score"],
                reverse=True,
            )

            # IMPORTANT:
            # For a general section query, return all
            # relevant product sections.
            evidence = evidence[
                :MAX_EVIDENCE_DOCUMENTS
            ]

        else:

            # No exact section match.
            # Fall back to normal semantic retrieval.
            evidence = candidates[
                :MAX_EVIDENCE_DOCUMENTS
            ]

    else:

        # ----------------------------------------------------
        # NAMED PRODUCT QUESTION
        # ----------------------------------------------------

        evidence = candidates[
            :MAX_EVIDENCE_DOCUMENTS
        ]

    # --------------------------------------------------------
    # FALLBACK IF EVIDENCE IS EMPTY
    # --------------------------------------------------------

    if not evidence:

        evidence = candidates[
            :MAX_EVIDENCE_DOCUMENTS
        ]

    elapsed = (
        time.perf_counter()
        - started
    )

    return {
        **state,

        "product_filter": product,

        "section_intent": intent,

        "documents": evidence,

        "metrics": {
            **state.get(
                "metrics",
                {},
            ),

            "retrieval_time": round(
                elapsed,
                4,
            ),

            "document_count": len(
                evidence
            ),

            "retrieved_products": list(
                dict.fromkeys(
                    str(
                        d.get(
                            "product",
                            "unknown",
                        )
                    )
                    for d in evidence
                )
            ),
        },
    }


# ============================================================
# CONTEXT BUILDER
# ============================================================

def build_context(
    documents: list[dict[str, Any]],
) -> str:

    blocks = []

    total = 0

    for idx, doc in enumerate(
        documents,
        1,
    ):

        block = (
            f"[C{idx}]\n"
            f"Product: {doc.get('product')}\n"
            f"Active ingredient: "
            f"{doc.get('active_ingredient')}\n"
            f"Section: {doc.get('section')}\n"
            f"Page: {doc.get('page')}\n"
            f"Source: {doc.get('source')}\n"
            f"Content:\n"
            f"{doc.get('content', '')}\n"
        )

        if (
            total + len(block)
            > MAX_CONTEXT_CHARS
        ):
            break

        blocks.append(block)

        total += len(block)

    return "\n---\n".join(blocks)


# ============================================================
# CITATION EXTRACTION
# ============================================================

def extract_citation_labels(
    answer: str,
) -> list[str]:

    labels = re.findall(
        r"\[C(\d+)\]",
        answer or "",
        flags=re.I,
    )

    seen = set()

    output = []

    for number in labels:

        label = f"C{number}"

        if label not in seen:

            seen.add(label)

            output.append(label)

    return output


def citation_labels_valid(
    answer: str,
    documents: list[dict[str, Any]],
) -> bool:

    labels = extract_citation_labels(
        answer
    )

    if not labels:
        return False

    valid = {
        f"C{i}"
        for i in range(
            1,
            len(documents) + 1,
        )
    }

    return all(
        label in valid
        for label in labels
    )


# ============================================================
# LLM
# ============================================================

def get_llm() -> ChatGroq:

    return ChatGroq(
        model=GROQ_MODEL,
        temperature=0,
        max_retries=2,
        timeout=30,
    )


# ============================================================
# GREETING PROMPT
# ============================================================

GREETING_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are the Renata PLC Leaflet Assistant.

The user is making a simple conversational greeting.

Respond naturally, politely, and briefly.

Do not discuss medicine unless the user asks about medicine.

Do not create citations.

Keep the response to one or two sentences.
Examples:

User: Hello
Assistant: Hello! How can I help you with the Renata PLC medicine leaflets?

User: Good morning
Assistant: Good morning! How can I help you with the Renata PLC leaflets today?
""",
        ),

        (
            "human",
            "{question}",
        ),
    ]
)


def generate_greeting(
    question: str,
) -> str:

    try:

        response = (
            GREETING_PROMPT
            | get_llm()
        ).invoke(
            {
                "question": question
            }
        )

        answer = str(
            response.content
        ).strip()

        if answer:
            return answer

    except Exception:
        pass

    return (
        "Hello! I'm the Renata PLC "
        "Leaflet Assistant. "
        "How can I help you today?"
    )


# ============================================================
# GENERATION PROMPT
# ============================================================

GENERATION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are the Renata PLC Leaflet Assistant.

The supplied evidence is the ONLY source of truth.

============================================================
CORE RULES
============================================================

1. Answer ONLY from the supplied evidence.

2. Never use outside medical knowledge.

3. Never invent facts.

4. Do not infer information that is not explicitly supported.

5. Do not call a medicine a tablet, capsule, syrup, etc.
   unless the supplied evidence explicitly says so.

6. Do not add strength, formulation, indication,
   contraindication, dosage, or other information
   unless supported by the evidence.

============================================================
CITATION RULES
============================================================

7. Every factual sentence or bullet MUST end with
   one or more citations.

Example:

Rolac contains ketorolac, a non-steroidal
anti-inflammatory drug (NSAID). [C1]

8. Never invent citation numbers.

9. Only use citation labels that exist in the evidence.

============================================================
PRODUCT RULES
============================================================

10. If the question names a product, answer ONLY
    for that product.

11. If the question does NOT name a product and the
    evidence contains multiple products, answer for
    EVERY relevant product represented in the evidence.

12. NEVER answer only using the first document.

13. For general questions involving a section such as:

    - pregnancy
    - breast-feeding
    - side effects
    - storage
    - precautions
    - indications
    - how to take

    organize the answer by PRODUCT when multiple
    products are present.

============================================================
IMPORTANT CROSS-PRODUCT RULE
============================================================

If the evidence contains:

[C1] Product A
[C2] Product B
[C3] Product C
[C4] Product D
[C5] Product E

and the question is:

"Use in pregnancy and breast-feeding"

then the answer MUST cover:

Product A [C1]
Product B [C2]
Product C [C3]
Product D [C4]
Product E [C5]

Do not stop after Product A.

============================================================
INSUFFICIENT EVIDENCE
============================================================

14. If the evidence genuinely does not contain enough
    information to answer the question, output exactly:

I can't answer that from the provided Renata leaflets.

============================================================
STYLE
============================================================

15. Keep the answer concise.

16. Prefer headings and bullets for multiple products.

17. Preserve the meaning of the leaflet.

18. Do not mention retrieval, vector databases,
    embeddings, context, prompts, or internal processing.

19. Do not provide general medical advice.

============================================================
QUESTION
============================================================
""",
        ),

        (
            "human",
            """
Question:

{question}

============================================================
SUPPLIED EVIDENCE
============================================================

{context}

============================================================
ANSWER
============================================================
""",
        ),
    ]
)


# ============================================================
# GENERATION NODE
# ============================================================

def generate_node(
    state: GraphState,
) -> GraphState:

    documents = state.get(
        "documents",
        [],
    )

    if not documents:
        return no_evidence_node(
            state
        )

    started = time.perf_counter()

    response = (
        GENERATION_PROMPT
        | get_llm()
    ).invoke(
        {
            "question": state[
                "question"
            ],

            "context": build_context(
                documents
            ),
        }
    )

    answer = str(
        response.content
    ).strip()

    return {
        **state,

        "answer": answer,

        "metrics": {
            **state.get(
                "metrics",
                {},
            ),

            "generation_time": round(
                time.perf_counter()
                - started,
                4,
            ),
        },
    }


# ============================================================
# VALIDATION PROMPT
# ============================================================

VALIDATION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a strict evidence validator for a pharmaceutical
leaflet assistant.

Return exactly one of:

SUPPORTED

or

UNSUPPORTED

============================================================
SUPPORTED
============================================================

Return SUPPORTED when:

- every factual claim in the answer is supported by
  the supplied evidence;

- citations point to evidence that supports the claim;

- the answer is a faithful paraphrase of the evidence;

- the answer does not materially change the meaning;

- the answer does not introduce outside medical knowledge.

Faithful paraphrasing IS allowed.

For example, if the evidence says:

"Rolac contains ketorolac, a non-steroidal
anti-inflammatory drug (NSAID)."

then this is supported:

"Rolac contains ketorolac, an NSAID."

============================================================
UNSUPPORTED
============================================================

Return UNSUPPORTED if:

- the answer adds a fact not present in the evidence;

- a citation does not support the associated claim;

- the answer changes the meaning of the leaflet;

- the answer gives medical advice not contained
  in the evidence;

- the answer claims a dosage, formulation,
  indication, warning, pregnancy statement, etc.
  that is not supported.

Do NOT reject an answer merely because it uses
different wording from the evidence.

Return ONLY:

SUPPORTED

or:

UNSUPPORTED
""",
        ),

        (
            "human",
            """
Question:

{question}

Evidence:

{context}

Answer:

{answer}

Decision:
""",
        ),
    ]
)


# ============================================================
# GROUNDING VALIDATION
# ============================================================

def validate_grounding(
    question: str,
    answer: str,
    docs: list[dict[str, Any]],
) -> bool:

    if (
        not answer.strip()
        or answer.strip() == ABSTAIN
    ):
        return False

    try:

        response = (
            VALIDATION_PROMPT
            | get_llm()
        ).invoke(
            {
                "question": question,

                "context": build_context(
                    docs
                ),

                "answer": answer,
            }
        )

        result = (
            str(
                response.content
            )
            .strip()
            .upper()
        )

        return result == "SUPPORTED"

    except Exception:

        return False


# ============================================================
# VALIDATION NODE
# ============================================================

def validate_node(
    state: GraphState,
) -> GraphState:

    answer = state.get(
        "answer",
        "",
    )

    documents = state.get(
        "documents",
        [],
    )

    citation_valid = (
        citation_labels_valid(
            answer,
            documents,
        )
    )

    grounded = False

    if citation_valid:

        grounded = validate_grounding(
            state["question"],
            answer,
            documents,
        )

    if (
        not citation_valid
        or not grounded
    ):

        return {
            **state,

            "answer": ABSTAIN,

            "citations": [],

            "grounded": False,

            "citation_valid": citation_valid,

            "abstained": True,
        }

    return {
        **state,

        "grounded": True,

        "citation_valid": True,

        "abstained": False,
    }


# ============================================================
# NO EVIDENCE
# ============================================================

def no_evidence_node(
    state: GraphState,
) -> GraphState:

    return {
        **state,

        "answer": ABSTAIN,

        "citations": [],

        "grounded": False,

        "citation_valid": False,

        "abstained": True,
    }


# ============================================================
# FINALIZE NODE
# ============================================================

def finalize_node(
    state: GraphState,
) -> GraphState:

    if state.get(
        "abstained"
    ):

        return {
            **state,
            "citations": [],
        }

    documents = state.get(
        "documents",
        [],
    )

    citations = []

    labels = extract_citation_labels(
        state.get(
            "answer",
            "",
        )
    )

    for label in labels:

        index = (
            int(
                label[1:]
            ) - 1
        )

        if (
            0
            <= index
            < len(documents)
        ):

            document = documents[
                index
            ]

            citations.append(
                {
                    "label": label,

                    "source": document.get(
                        "source",
                        "unknown",
                    ),

                    "page": document.get(
                        "page"
                    ),

                    "section": document.get(
                        "section",
                        "unknown",
                    ),

                    "product": document.get(
                        "product"
                    ),

                    "active_ingredient":
                        document.get(
                            "active_ingredient"
                        ),

                    "score": float(
                        document.get(
                            "score",
                            0.0,
                        )
                    ),
                }
            )

    return {
        **state,

        "citations": citations,

        "metrics": {
            **state.get(
                "metrics",
                {},
            ),

            "citation_count": len(
                citations
            ),
        },
    }


# ============================================================
# ROUTING
# ============================================================

def route_after_retrieval(
    state: GraphState,
) -> str:

    if state.get(
        "documents"
    ):
        return "generate"

    return "no_evidence"


# ============================================================
# BUILD GRAPH
# ============================================================

def build_graph():

    graph = StateGraph(
        GraphState
    )

    graph.add_node(
        "retrieve",
        retrieve_node,
    )

    graph.add_node(
        "generate",
        generate_node,
    )

    graph.add_node(
        "validate",
        validate_node,
    )

    graph.add_node(
        "finalize",
        finalize_node,
    )

    graph.add_node(
        "no_evidence",
        no_evidence_node,
    )

    graph.add_edge(
        START,
        "retrieve",
    )

    graph.add_conditional_edges(
        "retrieve",
        route_after_retrieval,
        {
            "generate": "generate",
            "no_evidence": "no_evidence",
        },
    )

    graph.add_edge(
        "generate",
        "validate",
    )

    graph.add_edge(
        "validate",
        "finalize",
    )

    graph.add_edge(
        "finalize",
        END,
    )

    graph.add_edge(
        "no_evidence",
        END,
    )

    return graph.compile()


# ============================================================
# GRAPH INSTANCE
# ============================================================

_graph = build_graph()


# ============================================================
# PUBLIC ASK FUNCTION
# ============================================================

def ask(
    question: str,
) -> dict[str, Any]:

    question = (
        question or ""
    ).strip()

    # --------------------------------------------------------
    # EMPTY QUESTION
    # --------------------------------------------------------

    if not question:

        return {
            "answer": "Please provide a question.",
            "citations": [],
            "metrics": {},
        }

    started = time.perf_counter()

    # --------------------------------------------------------
    # GREETING
    # --------------------------------------------------------

    if is_greeting(
        question
    ):

        answer = generate_greeting(
            question
        )

        return {
            "answer": answer,

            "citations": [],

            "metrics": {
                "total_time": round(
                    time.perf_counter()
                    - started,
                    4,
                ),

                "abstained": False,

                "greeting": True,

                "citation_count": 0,
            },
        }

    # --------------------------------------------------------
    # NORMAL RAG
    # --------------------------------------------------------

    result = _graph.invoke(
        {
            "question": question,
            "metrics": {},
        }
    )

    metrics = {
        **result.get(
            "metrics",
            {},
        ),

        "total_time": round(
            time.perf_counter()
            - started,
            4,
        ),

        "abstained": bool(
            result.get(
                "abstained",
                False,
            )
        ),

        "greeting": False,
    }

    return {
        "answer": result.get(
            "answer",
            ABSTAIN,
        ),

        "citations": result.get(
            "citations",
            [],
        ),

        "metrics": metrics,
    }


# ============================================================
# HEALTH CHECK
# ============================================================

def health_check() -> dict[str, Any]:

    from .config import CHROMA_DIR

    indexed = (
        CHROMA_DIR.exists()
        and any(
            CHROMA_DIR.iterdir()
        )
    )

    return {
        "status": (
            "ok"
            if indexed
            else "degraded"
        ),

        "indexed": indexed,

        "products": list(
            dict.fromkeys(
                PRODUCT_ALIASES.values()
            )
        ),
    }


# ============================================================
# PUBLIC API
# ============================================================

__all__ = [
    "ABSTAIN",
    "ask",
    "build_graph",
    "citation_labels_valid",
    "detect_product",
    "detect_product_filter",
    "detect_section_intent",
    "lexical_overlap",
    "extract_citation_labels",
    "normalize_text",
    "is_greeting",
    "health_check",
    "retrieve_node",
    "generate_node",
    "validate_node",
]