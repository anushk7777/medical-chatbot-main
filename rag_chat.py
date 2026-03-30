import re

import config


SYSTEM_PROMPT = (
    "You are MediBot, a medical information assistant. "
    "Answer using only the provided context. "
    "Write in clear, patient-friendly language. "
    "If the user asks for symptoms, causes, risk factors, or treatments, prefer 3 to 5 concise bullet points. "
    "If the user asks for a definition or explanation, answer in one short paragraph followed by one brief supporting paragraph when useful. "
    "If the context is incomplete, answer only what is supported and say what is missing in one short sentence. "
    "Do not mention that you were given context or source documents."
)


def retrieve_documents(vectorstore, prompt):
    try:
        return vectorstore.max_marginal_relevance_search(
            prompt,
            k=config.RETRIEVAL_K,
            fetch_k=config.RETRIEVAL_FETCH_K,
        )
    except Exception:
        return vectorstore.similarity_search(prompt, k=config.RETRIEVAL_K)


def build_context(source_documents):
    parts = []
    total_chars = 0

    for doc in source_documents:
        text = re.sub(r"\s+", " ", get_page_content(doc)).strip()
        if not text:
            continue

        metadata = get_metadata(doc)
        source = metadata.get("source", "Unknown source")
        page = metadata.get("page", "N/A")
        block = f"Source: {source} | Page: {page}\n{text}"

        if total_chars + len(block) > config.MAX_CONTEXT_CHARS:
            remaining = config.MAX_CONTEXT_CHARS - total_chars
            if remaining > 200:
                parts.append(block[:remaining])
            break

        parts.append(block)
        total_chars += len(block)

    return "\n\n".join(parts)


def get_page_content(doc):
    if isinstance(doc, dict):
        return doc.get("page_content", "")
    return getattr(doc, "page_content", "")


def get_metadata(doc):
    if isinstance(doc, dict):
        return doc.get("metadata", {}) or {}
    return getattr(doc, "metadata", {}) or {}


def _query_terms(prompt):
    stop_words = {
        "the", "and", "for", "with", "that", "this", "from", "into", "have", "what",
        "when", "where", "which", "who", "whom", "why", "how", "your", "about", "are",
        "was", "were", "will", "would", "could", "should", "there", "their", "them",
        "does", "doesnt", "don't", "dont", "tell", "explain", "please"
    }
    return {
        word for word in re.findall(r"[a-zA-Z0-9]+", prompt.lower())
        if len(word) > 2 and word not in stop_words
    }


def looks_like_model_refusal(answer_text):
    lowered = answer_text.lower().strip()
    refusal_markers = [
        "i don't know",
        "i dont know",
        "not directly stated",
        "not provided in the context",
        "not mentioned in the context",
        "cannot be determined from the context",
    ]
    return any(marker in lowered for marker in refusal_markers)


def extractive_fallback_answer(prompt, source_documents):
    query_terms = _query_terms(prompt)
    scored_sentences = []

    for doc in source_documents:
        cleaned_text = re.sub(r"\s+", " ", get_page_content(doc)).strip()
        if not cleaned_text:
            continue

        sentences = re.split(r"(?<=[.!?])\s+", cleaned_text)
        for sentence in sentences:
            sentence = sentence.strip(" -")
            if len(sentence) < 25:
                continue

            lowered = sentence.lower()
            if any(
                phrase in lowered for phrase in [
                    "not directly stated",
                    "not provided in the context",
                    "not mentioned in the context",
                    "cannot be determined",
                ]
            ):
                continue

            overlap = sum(1 for term in query_terms if term in lowered)
            if overlap == 0:
                continue

            score = overlap * 3
            if "definition" in lowered or "symptom" in lowered or "treatment" in lowered:
                score += 1
            if any(term in lowered[:80] for term in query_terms):
                score += 2
            if " is " in lowered or " are " in lowered or "refers to" in lowered or "characterized by" in lowered:
                score += 1
            if len(sentence) > 280:
                score -= 1

            scored_sentences.append((score, sentence))

    scored_sentences.sort(key=lambda item: item[0], reverse=True)

    selected = []
    for _, sentence in scored_sentences:
        if sentence not in selected:
            selected.append(sentence)
        if len(selected) == 2:
            break

    if selected:
        return " ".join(selected)

    if source_documents:
        cleaned_text = re.sub(r"\s+", " ", get_page_content(source_documents[0])).strip()
        sentences = [s.strip(" -") for s in re.split(r"(?<=[.!?])\s+", cleaned_text) if s.strip()]
        return " ".join(sentences[:2])

    return None


def generate_chat_answer(prompt, source_documents, client, model_name):
    context = build_context(source_documents)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Question: {prompt}\n\nContext:\n{context}",
        },
    ]

    completion = client.chat_completion(
        model=model_name,
        messages=messages,
        max_tokens=350,
        temperature=0.2,
    )
    answer = completion.choices[0].message.content.strip()

    if not answer or looks_like_model_refusal(answer):
        fallback = extractive_fallback_answer(prompt, source_documents)
        if fallback:
            return fallback

    return answer
