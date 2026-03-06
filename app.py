"""
app.py - Smart Document Assistant
Run: python app.py
Open: http://127.0.0.1:7860
"""

import gradio as gr
from src.ingestion import ingest
from src.retrieval import answer, summarize, reload_vectorstore

last_file = {"name": None}


def upload_file(file):
    if file is None:
        return "Please select a file first.", ""
    try:
        import src.retrieval as _ret
        import gc
        _ret._vectorstore = None
        gc.collect()

        from src.evaluation import reset_questions
        reset_questions()

        stats = ingest(file.name)
        last_file["name"] = stats["file_name"]
        reload_vectorstore()

        msg = (
            f"✅ File ingested successfully!\n\n"
            f"File: {stats['file_name']}\n"
            f"Characters: {stats['characters']:,}\n"
            f"Chunks: {stats['chunks']}\n\n"
            f"Go to the Chat tab and start asking questions."
        )
        return msg, stats["file_name"]
    except Exception as e:
        return f"❌ Error: {e}", ""


def chat_start(message, history, current_file):
    """Shows loading message immediately."""
    if not message.strip():
        return history, "", ""
    history = history + [(message, "⏳ Thinking...")]
    return history, "", message


def chat_respond(pending_msg, history, current_file):
    """Generates the actual response."""
    if not pending_msg:
        return history, ""

    if not (current_file or last_file["name"]):
        history[-1] = (history[-1][0], "Please upload a document first.")
        return history, ""

    try:
        from src.evaluation import record_question
        record_question(pending_msg)
        response, _ = answer(pending_msg, history[:-1])
    except Exception as e:
        response = f"❌ Error: {e}"

    history[-1] = (history[-1][0], response)
    return history, ""


def get_summary(current_file):
    name = current_file or last_file["name"]
    if not name:
        return "Please upload a document first."
    try:
        return "⏳ Generating summary, please wait..."
    except Exception as e:
        return f"❌ Error: {e}"


def get_summary_result(current_file):
    name = current_file or last_file["name"]
    if not name:
        return "Please upload a document first."
    try:
        return summarize(name)
    except Exception as e:
        return f"❌ Error: {e}"


def run_eval(current_file):
    name = current_file or last_file["name"]
    if not name:
        return "Please upload a document first.", None
    try:
        from src.evaluation import run_full_evaluation
        markdown, pdf_path = run_full_evaluation()
        return markdown, pdf_path
    except Exception as e:
        return f"❌ Error: {e}", None


with gr.Blocks(title="Smart Contract Assistant") as demo:

    current_file = gr.State(value="")
    pending_msg  = gr.State(value="")

    gr.HTML("""
    <div style="text-align:center; padding:20px 0 10px">
        <h1 style="font-size:2rem; color:#1e40af">📄 Smart Contract Assistant</h1>
        <p style="color:#64748b">Upload a contract and ask questions about it</p>
    </div>
    """)

    with gr.Tabs():

        # Tab 1: Upload
        with gr.Tab("📤 Upload Document"):
            gr.Markdown("### Upload a PDF or DOCX contract")
            file_input    = gr.File(label="Select file", file_types=[".pdf", ".docx"])
            upload_btn    = gr.Button("Upload & Process", variant="primary")
            upload_status = gr.Textbox(label="Status", value="Select a file and click Upload.", lines=6, interactive=False)
            upload_btn.click(fn=upload_file, inputs=[file_input], outputs=[upload_status, current_file])

        # Tab 2: Chat
        with gr.Tab("💬 Chat"):
            gr.Markdown("### Ask questions about the contract")
            gr.Markdown("*Guardrails active — uses semantic similarity to block off-topic questions.*")
            chatbot  = gr.Chatbot(height=420)
            with gr.Row():
                msg_box  = gr.Textbox(placeholder="Type your question here...", scale=5, lines=2, label="")
                send_btn = gr.Button("Send", variant="primary", scale=1)
            clear_btn = gr.Button("Clear chat")

            gr.Examples(
                examples=[
                    ["What is this contract about?"],
                    ["Who are the parties involved?"],
                    ["What are the main obligations?"],
                    ["What are the payment terms?"],
                    ["How can this contract be terminated?"],
                ],
                inputs=[msg_box],
            )

            # Two-step: show loading first, then respond
            send_btn.click(
                chat_start, [msg_box, chatbot, current_file], [chatbot, msg_box, pending_msg]
            ).then(
                chat_respond, [pending_msg, chatbot, current_file], [chatbot, pending_msg]
            )
            msg_box.submit(
                chat_start, [msg_box, chatbot, current_file], [chatbot, msg_box, pending_msg]
            ).then(
                chat_respond, [pending_msg, chatbot, current_file], [chatbot, pending_msg]
            )
            clear_btn.click(lambda: ([], ""), outputs=[chatbot, msg_box])

        # Tab 3: Summary
        with gr.Tab("📋 Summary"):
            gr.Markdown("### Auto-generated contract summary")
            summarize_btn = gr.Button("Generate Summary", variant="primary")
            summary_out   = gr.Markdown("*Click the button after uploading a contract.*")

            summarize_btn.click(
                lambda cf: "⏳ Generating summary, please wait...",
                inputs=[current_file],
                outputs=[summary_out]
            ).then(
                get_summary_result,
                inputs=[current_file],
                outputs=[summary_out]
            )

        # Tab 4: Evaluation
        with gr.Tab("📊 Evaluation"):
            gr.Markdown("### RAG Pipeline Evaluation")
            gr.Markdown("Uses **LLM-as-Judge** to score answer quality + semantic retrieval metrics. Downloads a PDF report.")
            eval_btn     = gr.Button("▶ Run Evaluation & Download Report", variant="primary")
            eval_out     = gr.Markdown("*Ask questions in Chat first, then click Run Evaluation.*")
            pdf_download = gr.File(label="📥 Download Markdown Report", visible=False)

            def run_and_show(current_file):
                md, pdf = run_eval(current_file)
                return md, gr.File.update(value=pdf, visible=pdf is not None)

            eval_btn.click(
                lambda cf: ("⏳ Running evaluation, please wait...", None),
                inputs=[current_file], outputs=[eval_out, pdf_download]
            ).then(
                run_and_show, inputs=[current_file], outputs=[eval_out, pdf_download]
            )

    gr.HTML("""
    <div style="text-align:center; padding:10px; color:#94a3b8; font-size:0.8rem">
        Powered by Groq + LangChain + Gradio · All data stored locally
    </div>
    """)


if __name__ == "__main__":
    demo.launch(server_port=7860, show_error=True)
