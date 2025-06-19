import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from langchain_community.llms import Ollama
import gradio as gr
import warnings

warnings.filterwarnings('ignore')


class CSVQASystem:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.llm = Ollama(
            model="phi3",
            temperature=0.3,
            top_k=50,
            top_p=0.9,
            repeat_penalty=1.1
        )
        self.text_data = None
        self.vectorstore = None

    def load_csv(self, file_path: str, sep: str = ";") -> str:
        try:
            df = pd.read_csv(file_path, sep=sep, encoding="utf-8")
        except UnicodeDecodeError:
            try:
                df = pd.read_csv(file_path, sep=sep, encoding="utf-16")
            except UnicodeDecodeError:
                df = pd.read_csv(file_path, sep=sep, encoding="ISO-8859-1")

        # Remove colunas vazias
        df.dropna(axis=1, how='all', inplace=True)

        # Substitui valores ausentes por string vazia
        df.fillna("", inplace=True)

        if df.empty:
            return "❌ O CSV está vazio após limpeza. Verifique o conteúdo."

        self.text_data = df.to_string(index=False)

        texts = []
        for _, row in df.iterrows():
            txt = row.to_string()
            if isinstance(txt, str) and txt.strip() != "":
                texts.append(txt)

        if not texts:
            return "❌ Nenhum dado válido foi encontrado para vetorização."

        docs = [Document(page_content=txt) for txt in texts]
        self.vectorstore = FAISS.from_documents(docs, self.embeddings)

        return "✅ Arquivo CSV carregado e vetorizado com sucesso!"

    def ask(self, question: str) -> str:
        if not self.text_data:
            return "❌ Nenhum CSV carregado ainda."

        retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})
        context_docs = retriever.get_relevant_documents(question)
        context = "\n\n".join([doc.page_content for doc in context_docs])

        prompt = f"""Você é um especialista em análise de dados. Com base nos dados abaixo, responda à pergunta de forma objetiva, clara e precisa.

### DADOS:
{context}

### PERGUNTA:
{question}

### RESPOSTA:"""

        resposta = self.llm.invoke(prompt)
        return resposta.strip()


qa_system = CSVQASystem()


# Funções para Gradio
def upload_csv(file):
    try:
        return qa_system.load_csv(file.name)
    except Exception as e:
        return f"❌ Erro ao carregar o arquivo: {str(e)}"

def responder(pergunta):
    try:
        return qa_system.ask(pergunta)
    except Exception as e:
        return f"❌ Erro ao processar a pergunta: {str(e)}"


# Interface Gradio estilo ChatGPT
with gr.Blocks(title="Chat com CSV") as interface:
    gr.Markdown("# 🤖 Chat com seu CSV\nEnvie um arquivo CSV e faça perguntas sobre ele.")

    with gr.Row():
        file_input = gr.File(label="📎 Envie seu arquivo CSV", file_types=[".csv"], type="filepath")
        file_status = gr.Textbox(label="📄 Status do arquivo", interactive=False)

    file_input.change(fn=upload_csv, inputs=file_input, outputs=file_status)

    chatbot = gr.Chatbot(label="💬 Assistente de Dados")

    with gr.Row():
        msg = gr.Textbox(placeholder="Digite sua pergunta sobre o CSV...")
        send_btn = gr.Button("Enviar")

    def chat_func(history, user_input):
        resposta = responder(user_input)
        history.append((user_input, resposta))
        return history, ""

    send_btn.click(chat_func, inputs=[chatbot, msg], outputs=[chatbot, msg])
    msg.submit(chat_func, inputs=[chatbot, msg], outputs=[chatbot, msg])

interface.launch()
