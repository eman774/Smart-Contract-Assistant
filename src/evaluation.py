"""
evaluation.py
=============
Evaluates RAG pipeline quality:
- Retrieval quality: hit rate, speed, chunk coverage
- Answer quality: relevance, groundedness, citation (LLM-as-Judge)
Exports a professional readable PDF report.
"""

import time
import os
from datetime import datetime
from src.retrieval import get_vectorstore, answer
from src.config import TOP_K

_asked_questions = []

FALLBACK_QUESTIONS = [
    "What is this contract about?",
    "Who are the parties involved in this contract?",
    "What are the main obligations of each party?",
    "What are the payment terms?",
    "How can this contract be terminated?",
]


def reset_questions():
    global _asked_questions
    _asked_questions = []


def record_question(question: str):
    if question.strip() and question not in _asked_questions:
        _asked_questions.append(question.strip())


def get_questions_to_evaluate() -> list:
    return _asked_questions if _asked_questions else FALLBACK_QUESTIONS


def llm_judge(question: str, answer_text: str, context: str) -> dict:
    from src.retrieval import get_llm

    prompt = f"""You are an expert evaluator for a RAG system.

Question: {question}

Document Context:
{context[:800]}

Answer:
{answer_text[:500]}

Reply ONLY with this exact format:
RELEVANCE: [0-100]
GROUNDEDNESS: [0-100]
REASONING: [one sentence]

RELEVANCE = does the answer address the question (100 = perfect)
GROUNDEDNESS = is the answer based on the document (100 = fully grounded)"""

    try:
        result    = get_llm(max_tokens=100).invoke(prompt)
        text      = result.content.strip()
        relevance = 50
        grounding = 50
        reasoning = "LLM evaluation completed."

        for line in text.split("\n"):
            line = line.strip()
            if line.startswith("RELEVANCE:"):
                try: relevance = int(line.split(":")[1].strip())
                except: pass
            elif line.startswith("GROUNDEDNESS:"):
                try: grounding = int(line.split(":")[1].strip())
                except: pass
            elif line.startswith("REASONING:"):
                reasoning = line.split(":", 1)[1].strip()

        return {
            "relevance": min(max(relevance, 0), 100),
            "grounding": min(max(grounding, 0), 100),
            "reasoning": reasoning,
            "method":    "LLM-as-Judge",
        }

    except Exception as e:
        q_words   = set(question.lower().split()) - {"what","who","when","where","how","is","are","the","a","an","this","contract"}
        a_lower   = answer_text.lower()
        matches   = sum(1 for w in q_words if w in a_lower)
        rel       = min(round(matches / max(len(q_words), 1) * 100), 100)
        ctx_words = set(context.lower().split())
        ans_words = set(answer_text.lower().split())
        overlap   = len(ctx_words & ans_words)
        gnd       = min(round(overlap / max(len(ans_words), 1) * 100), 100)
        return {
            "relevance": rel,
            "grounding": gnd,
            "reasoning": f"Heuristic fallback (LLM judge failed: {e})",
            "method":    "Heuristic",
        }


def score_answer(question: str, answer_text: str, context: str) -> dict:
    scores       = llm_judge(question, answer_text, context)
    has_citation = "Source:" in answer_text or "📄" in answer_text
    not_found    = "not found in the document" in answer_text.lower()
    length       = len(answer_text)

    if not_found:       completeness = "Not found in document"
    elif length < 50:   completeness = "Too short"
    elif length < 500:  completeness = "Good"
    else:               completeness = "Detailed"

    return {
        "relevance":    scores["relevance"],
        "grounding":    scores["grounding"],
        "reasoning":    scores["reasoning"],
        "method":       scores["method"],
        "has_citation": has_citation,
        "completeness": completeness,
        "not_found":    not_found,
        "length":       length,
    }


def run_full_evaluation() -> tuple:
    """Main evaluation function. Returns (markdown, md_path)."""
    questions      = get_questions_to_evaluate()
    using_fallback = len(_asked_questions) == 0
    now            = datetime.now().strftime("%Y-%m-%d %H:%M")

    vs        = get_vectorstore()
    retriever = vs.as_retriever(search_kwargs={"k": TOP_K})
    results   = []

    for q in questions:
        t0      = time.time()
        docs    = retriever.invoke(q)
        ret_time = round(time.time() - t0, 3)
        context  = "\n".join([d.page_content for d in docs])

        t1 = time.time()
        try:
            ans_text, _ = answer(q, [])
        except Exception as e:
            ans_text = f"Error: {e}"
        ans_time = round(time.time() - t1, 2)

        quality = score_answer(q, ans_text, context)
        results.append({
            "question":      q,
            "hit":           len(docs) > 0,
            "chunks":        len(docs),
            "avg_chunk_len": round(sum(len(d.page_content) for d in docs) / len(docs)) if docs else 0,
            "ret_time":      ret_time,
            "ans_time":      ans_time,
            "answer":        ans_text,
            "quality":       quality,
        })

    hit_rate      = sum(1 for r in results if r["hit"]) / len(results) * 100
    citation_rate = sum(1 for r in results if r["quality"]["has_citation"]) / len(results) * 100
    avg_relevance = sum(r["quality"]["relevance"] for r in results) / len(results)
    avg_grounding = sum(r["quality"]["grounding"] for r in results) / len(results)
    avg_ret_time  = sum(r["ret_time"] for r in results) / len(results)
    avg_ans_time  = sum(r["ans_time"] for r in results) / len(results)

    md       = _build_markdown(results, hit_rate, citation_rate, avg_relevance,
                               avg_grounding, avg_ret_time, avg_ans_time, now, using_fallback)

    # Save as .md file
    os.makedirs("./data", exist_ok=True)
    md_path = "./data/evaluation_report.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md)

    return md, md_path


def evaluate_retrieval_only() -> str:
    """Alias used by server.py — runs full evaluation and returns markdown only."""
    md, _ = run_full_evaluation()
    return md


def _build_markdown(results, hit_rate, citation_rate, avg_relevance, avg_grounding,
                    avg_ret_time, avg_ans_time, now, using_fallback) -> str:
    lines = [f"# RAG Evaluation Report\n*{now}*\n"]

    if using_fallback:
        lines.append("*Using preset questions. Ask questions in Chat first for personalized results.*\n")
    else:
        lines.append(f"*Based on {len(results)} question(s) from your session.*\n")

    lines += [
        "## Summary",
        f"| Metric | Value | Meaning |",
        f"|---|---|---|",
        f"| Hit Rate | {hit_rate:.0f}% | Questions with relevant chunks found |",
        f"| Citation Rate | {citation_rate:.0f}% | Answers with source references |",
        f"| Avg Relevance | {avg_relevance:.0f}% | How well answers matched questions (LLM-as-Judge) |",
        f"| Avg Groundedness | {avg_grounding:.0f}% | How much answers were based on document |",
        f"| Avg Retrieval Time | {avg_ret_time:.3f}s | Time to find relevant chunks |",
        f"| Avg Answer Time | {avg_ans_time:.2f}s | Total time to generate answer |",
        "",
        "## Per-Question Results",
    ]

    for i, r in enumerate(results, 1):
        q = r["quality"]
        lines.append(f"### Q{i}: {r['question']}")
        lines.append(f"- **Chunks:** {r['chunks']}/{TOP_K} | **Retrieval:** {r['ret_time']}s | **Answer time:** {r['ans_time']}s")
        lines.append(f"- **Relevance:** {q['relevance']}% | **Groundedness:** {q['grounding']}% | **Citation:** {'✅' if q['has_citation'] else '❌'} | **Quality:** {q['completeness']}")
        lines.append(f"- **Judge ({q.get('method','?')}):** {q.get('reasoning','')}")
        lines.append(f"- **Answer:** *{r['answer'][:250].replace(chr(10), ' ')}...*")
        lines.append("")

    lines += [
        "## Known Limitations",
        "- Single document at a time",
        "- English contracts only",
        "- LLM response time varies with free API tier",
        "- LLM-as-Judge scores are approximations",
        "- Guardrails use semantic similarity with fixed threshold",
    ]

    return "\n".join(lines)


def _build_pdf(results, hit_rate, citation_rate, avg_relevance, avg_grounding,
               avg_ret_time, avg_ans_time, now, using_fallback) -> str:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable, KeepTogether
    from reportlab.lib.units import cm

    os.makedirs("./data", exist_ok=True)
    path = "./data/evaluation_report.pdf"
    doc  = SimpleDocTemplate(path, pagesize=A4,
                             leftMargin=3*cm, rightMargin=3*cm,
                             topMargin=3*cm, bottomMargin=3*cm)

    BLUE  = colors.HexColor("#1e40af")
    GRAY  = colors.HexColor("#6b7280")
    GREEN = colors.HexColor("#16a34a")
    RED   = colors.HexColor("#dc2626")
    AMBER = colors.HexColor("#d97706")

    title_s  = ParagraphStyle("title",  fontSize=20, fontName="Helvetica-Bold", textColor=BLUE, spaceAfter=4)
    sub_s    = ParagraphStyle("sub",    fontSize=10, textColor=GRAY, spaceAfter=2)
    date_s   = ParagraphStyle("date",   fontSize=9,  textColor=GRAY, spaceAfter=16)
    h1_s     = ParagraphStyle("h1",     fontSize=13, fontName="Helvetica-Bold", textColor=BLUE, spaceBefore=20, spaceAfter=8)
    h2_s     = ParagraphStyle("h2",     fontSize=11, fontName="Helvetica-Bold", textColor=colors.HexColor("#1e3a8a"), spaceBefore=14, spaceAfter=4)
    body_s   = ParagraphStyle("body",   fontSize=10, leading=16, spaceAfter=6)
    metric_s = ParagraphStyle("metric", fontSize=10, leading=16, spaceAfter=4, leftIndent=20)
    note_s   = ParagraphStyle("note",   fontSize=9,  textColor=GRAY, leading=14, spaceAfter=4)
    answer_s = ParagraphStyle("ans",    fontSize=9,  textColor=colors.HexColor("#374151"),
                               leading=14, leftIndent=15, spaceAfter=6)

    def rating(val, good=80, ok=60):
        if val >= good: return ("Good",               GREEN)
        if val >= ok:   return ("Fair",               AMBER)
        return              ("Needs Improvement",    RED)

    story = []

    # Title
    story.append(Paragraph("RAG Pipeline Evaluation Report", title_s))
    story.append(Paragraph("Smart Contract Assistant", sub_s))
    story.append(Paragraph(f"Generated: {now}", date_s))
    story.append(HRFlowable(width="100%", thickness=1.5, color=BLUE, spaceAfter=16))

    if using_fallback:
        story.append(Paragraph(
            "Note: Using preset test questions. Ask questions in Chat first for a personalized report.", note_s))
    else:
        story.append(Paragraph(
            f"This report is based on {len(results)} question(s) asked during this session.", note_s))
    story.append(Spacer(1, 8))

    # About
    story.append(Paragraph("About This Report", h1_s))
    story.append(Paragraph(
        "This report evaluates the performance of the RAG pipeline used in the Smart Contract Assistant. "
        "It measures two things:", body_s))
    story.append(Paragraph(
        "<b>Retrieval Quality:</b>  Did the system find the right parts of the contract to answer each question?",
        metric_s))
    story.append(Paragraph(
        "<b>Answer Quality:</b>  Were the answers accurate and grounded in the document? "
        "Scored using LLM-as-Judge methodology — the LLM evaluates its own answers.", metric_s))
    story.append(Spacer(1, 6))

    # Summary
    story.append(Paragraph("Summary", h1_s))
    metrics = [
        ("Hit Rate",           f"{hit_rate:.0f}%",      "% of questions where the retriever found relevant chunks"),
        ("Citation Rate",      f"{citation_rate:.0f}%", "% of answers that included a source reference"),
        ("Avg Relevance",      f"{avg_relevance:.0f}%", "How well answers matched the questions (LLM-as-Judge)"),
        ("Avg Groundedness",   f"{avg_grounding:.0f}%", "How much answers were based on document content"),
        ("Avg Retrieval Time", f"{avg_ret_time:.3f}s",  "Time to retrieve relevant document chunks"),
        ("Avg Answer Time",    f"{avg_ans_time:.2f}s",  "Total time to generate a complete answer"),
    ]
    for name, value, meaning in metrics:
        story.append(Paragraph(f"<b>{name}:</b>  {value}  —  {meaning}", metric_s))

    story.append(Spacer(1, 6))
    story.append(Paragraph(
        "A Hit Rate of 100% means the retriever always found relevant sections. "
        "Relevance and Groundedness above 70% indicate good answer quality.", note_s))

    # Per-question
    story.append(Paragraph("Question-by-Question Results", h1_s))

    for i, r in enumerate(results, 1):
        q      = r["quality"]
        status = "PASS" if r["hit"] else "FAIL"
        s_col  = GREEN if r["hit"] else RED
        rel_r, _  = rating(q["relevance"])
        gnd_r, _  = rating(q["grounding"])

        block = []
        block.append(Paragraph(f"Question {i}  [{status}]", h2_s))
        block.append(Paragraph(f'"{r["question"]}"', body_s))
        block.append(Paragraph(
            f"<b>Retrieval:</b>  {r['chunks']}/{TOP_K} chunks  |  "
            f"Retrieval time: {r['ret_time']}s  |  Answer time: {r['ans_time']}s", metric_s))
        block.append(Paragraph(
            f"<b>Relevance Score:</b>  {q['relevance']}%  ({rel_r})  —  "
            f"How well the answer addressed the question", metric_s))
        block.append(Paragraph(
            f"<b>Groundedness Score:</b>  {q['grounding']}%  ({gnd_r})  —  "
            f"How much the answer was based on the document", metric_s))
        block.append(Paragraph(
            f"<b>Source Citation:</b>  {'Yes' if q['has_citation'] else 'No'}  |  "
            f"<b>Answer Quality:</b>  {q['completeness']}  |  "
            f"<b>Scoring:</b>  {q.get('method','?')}", metric_s))

        reasoning = q.get("reasoning", "")
        if reasoning:
            block.append(Paragraph(f"<b>Judge Reasoning:</b>  {reasoning}", note_s))

        preview = r["answer"][:350].replace("\n", " ")
        if len(r["answer"]) > 350:
            preview += "..."
        block.append(Spacer(1, 4))
        block.append(Paragraph("<b>Answer:</b>", note_s))
        block.append(Paragraph(preview, answer_s))
        block.append(HRFlowable(width="100%", thickness=0.3,
                                color=colors.HexColor("#e5e7eb"), spaceAfter=8))
        story.append(KeepTogether(block))

    # Limitations
    story.append(Paragraph("Known Limitations", h1_s))
    limitations = [
        ("Single document only",   "The system handles one document at a time. A new upload replaces the previous one."),
        ("English contracts only", "The pipeline is optimized for English. Other languages may produce poor results."),
        ("Free API rate limits",   "The Groq free tier has daily limits that may affect speed and availability."),
        ("LLM-as-Judge accuracy",  "Quality scores are generated by the LLM itself and are approximations, not ground truth."),
        ("Semantic guardrails",    "Off-topic detection uses embedding similarity with a fixed threshold."),
        ("Not production-ready",   "This is a prototype and should not be used for legal or compliance decisions."),
    ]
    for name, desc in limitations:
        story.append(Paragraph(f"<b>{name}:</b>  {desc}", metric_s))

    doc.build(story)
    return path
