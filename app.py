import sys
import os
import json
import gc
from typing import List, Dict, Any
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                               QHBoxLayout, QTextEdit, QLineEdit, QPushButton, 
                               QFileDialog, QLabel, QGroupBox, QSpinBox, QMessageBox)
from PySide6.QtCore import QThread, Signal, Qt
from PySide6.QtGui import QFont

# --- External Libraries ---
from llama_cpp import Llama
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from chromadb import Client as ChromaClient
from chromadb.config import Settings
from duckduckgo_search import DDGS

# ==========================================================
# 1. MEDICAL TOOLS, RAG, & SEARCH (Same as before)
# ==========================================================
CREDIBLE_MEDICAL_SITES = (
    "site:webmd.com OR site:mayoclinic.org OR site:nih.gov OR "
    "site:cdc.gov OR site:hopkinsmedicine.org OR site:who.int OR "
    "site:ncbi.nlm.nih.gov OR site:my.clevelandclinic.org"
)

def calculate_bmi(weight_kg: float, height_cm: float) -> str:
    height_m = height_cm / 100.0
    if height_m == 0: return "Error: Height cannot be zero."
    bmi = weight_kg / (height_m ** 2)
    category = "Underweight" if bmi < 18.5 else "Normal weight" if bmi < 25 else "Overweight" if bmi < 30 else "Obese"
    return f"BMI: {bmi:.1f} ({category})"

def calculate_gcs(eye: int, verbal: int, motor: int) -> str:
    if not (1 <= eye <= 4) or not (1 <= verbal <= 5) or not (1 <= motor <= 6):
        return "Invalid GCS inputs."
    total = eye + verbal + motor
    severity = "Mild" if total >= 13 else "Moderate" if total >= 9 else "Severe"
    return f"GCS: {total}/15 ({severity} Trauma)"

class MedicalRAG:
    def __init__(self):
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.chroma = ChromaClient(Settings(anonymized_telemetry=False))
        self.collection = self.chroma.get_or_create_collection(name="medical_docs")

    def ingest_pdf(self, file_path: str):
        reader = PdfReader(file_path)
        text = "".join([p.extract_text() + "\n" for p in reader.pages])
        chunks = [text[i:i+500] for i in range(0, len(text), 500)]
        chunks = [c for c in chunks if len(c.strip()) > 50]
        embeddings = self.embedder.encode(chunks).tolist()
        ids = [f"doc_{os.path.basename(file_path)}_{i}" for i in range(len(chunks))]
        self.collection.add(embeddings=embeddings, documents=chunks, ids=ids)
        return len(chunks)

    def query_rag(self, query: str) -> str:
        if self.collection.count() == 0: return "No PDFs loaded."
        query_embedding = self.embedder.encode([query]).tolist()
        results = self.collection.query(query_embeddings=query_embedding, n_results=3)
        return "\n---\n".join(results['documents'][0])

def web_search(query: str) -> str:
    try:
        with DDGS() as ddgs:
            restricted_query = f"{query} {CREDIBLE_MEDICAL_SITES}"
            results = list(ddgs.text(restricted_query, max_results=3))
            if not results:
                results = list(ddgs.text(query, max_results=3))
                if not results: return "No medical information found."
                prefix = "[General Web Fallback]\n\n"
            else:
                prefix = "[Credible Medical Sources]\n\n"
            return prefix + "\n\n".join([f"Title: {r['title']}\nSnippet: {r['body']}" for r in results])
    except Exception as e:
        return f"Web search failed: {str(e)}"

TOOL_MAP = {
    "calculate_bmi": calculate_bmi,
    "calculate_gcs": calculate_gcs,
    "search_medical_pdf": lambda kwargs: rag_system.query_rag(kwargs["query"]),
    "web_search": lambda kwargs: web_search(kwargs["query"])
}

TOOLS = [
    {"type": "function", "function": {"name": "calculate_bmi", "description": "Calculates BMI.", "parameters": {"type": "object", "properties": {"weight_kg": {"type": "number"}, "height_cm": {"type": "number"}}, "required": ["weight_kg", "height_cm"]}}},
    {"type": "function", "function": {"name": "calculate_gcs", "description": "Calculates Glasgow Coma Scale.", "parameters": {"type": "object", "properties": {"eye": {"type": "integer"}, "verbal": {"type": "integer"}, "motor": {"type": "integer"}}, "required": ["eye", "verbal", "motor"]}}},
    {"type": "function", "function": {"name": "search_medical_pdf", "description": "Searches uploaded local PDFs.", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "web_search", "description": "Searches WebMD, Mayo Clinic, NIH, CDC.", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}}
]

rag_system = MedicalRAG()

# ==========================================================
# 2. INFERENCE THREAD (Now receives the model instance)
# ==========================================================
class LLMThread(QThread):
    response_chunk = Signal(str)
    finished_signal = Signal()
    tool_call_signal = Signal(str)

    def __init__(self, llm_instance: Llama, messages: List[Dict]):
        super().__init__()
        self.llm = llm_instance
        self.messages = messages

    def run(self):
        if self.messages[0]["role"] != "system":
            self.messages.insert(0, {"role": "system", "content": "You are an AI Medical Assistant. Use tools for calculations and searches. Remind users this is not professional medical advice."})

        max_loops = 5
        loop_count = 0

        while loop_count < max_loops:
            loop_count += 1
            try:
                response = self.llm.create_chat_completion(
                    messages=self.messages, tools=TOOLS, tool_choice="auto", stream=True
                )
            except Exception as e:
                self.response_chunk.emit(f"\n[Error: {str(e)}]")
                break

            tool_calls = []
            current_content = ""
            
            for chunk in response:
                delta = chunk["choices"][0]["delta"]
                if "content" in delta:
                    current_content += delta["content"]
                    self.response_chunk.emit(delta["content"])
                if "tool_calls" in delta:
                    for tc in delta["tool_calls"]:
                        if tc.get("function"):
                            if len(tool_calls) <= tc["index"]:
                                tool_calls.append({"id": tc["id"], "function": {"name": tc["function"]["name"], "arguments": ""}})
                            tool_calls[tc["index"]]["function"]["arguments"] += tc["function"].get("arguments", "")

            if not tool_calls: break

            self.messages.append({"role": "assistant", "content": current_content if current_content else None, "tool_calls": tool_calls})
            
            for tc in tool_calls:
                func_name = tc["function"]["name"]
                try: func_args = json.loads(tc["function"]["arguments"])
                except: func_args = {}

                self.tool_call_signal.emit(f"⏳ Running: {func_name}...")
                result = TOOL_MAP[func_name](func_args) if func_name in TOOL_MAP else "Unknown tool."
                
                self.messages.append({"role": "tool", "tool_call_id": tc["id"], "content": str(result)})
                self.tool_call_signal.emit(f"✅ Result received.\n")

        self.finished_signal.emit()

# ==========================================================
# 3. PYSIDE6 GUI WITH GPU/RESOURCE MANAGEMENT
# ==========================================================
class MedicalAssistantGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Medical Assistant - GPU Manager")
        self.setGeometry(100, 100, 900, 750)
        
        # Core State Variables for Resource Management
        self.llm = None
        self.is_generating = False
        self.chat_history = []

        self.init_ui()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # --- GPU & MODEL MANAGEMENT PANEL ---
        gpu_group = QGroupBox("Model & GPU Resource Management")
        gpu_layout = QVBoxLayout()
        
        # Model Path
        path_layout = QHBoxLayout()
        self.model_path_input = QLineEdit()
        self.model_path_input.setPlaceholderText("Path to .gguf model file...")
        self.browse_btn = QPushButton("Browse")
        self.browse_btn.clicked.connect(self.browse_model)
        path_layout.addWidget(self.model_path_input)
        path_layout.addWidget(self.browse_btn)
        gpu_layout.addLayout(path_layout)

        # Settings Row (GPU Layers & Context)
        settings_layout = QHBoxLayout()
        
        settings_layout.addWidget(QLabel("GPU Layers (-1 for All):"))
        self.gpu_layers_spin = QSpinBox()
        self.gpu_layers_spin.setRange(-1, 99)
        self.gpu_layers_spin.setValue(-1)
        self.gpu_layers_spin.setToolTip("-1 = Offload everything to GPU (Most VRAM)\n0 = Run on CPU only (No VRAM)\n1-99 = Split load between CPU/GPU")
        settings_layout.addWidget(self.gpu_layers_spin)

        settings_layout.addWidget(QLabel("Context Window:"))
        self.ctx_spin = QSpinBox()
        self.ctx_spin.setRange(512, 16384)
        self.ctx_spin.setSingleStep(512)
        self.ctx_spin.setValue(4096)
        self.ctx_spin.setToolTip("Size of conversation memory. Larger = uses much more VRAM.")
        settings_layout.addWidget(self.ctx_spin)

        settings_layout.addStretch()
        gpu_layout.addLayout(settings_layout)

        # Load/Unload Buttons
        btn_layout = QHBoxLayout()
        self.load_btn = QPushButton("⚡ Load Model into Memory")
        self.load_btn.setStyleSheet("background-color: #28a745; color: white; font-weight: bold; padding: 8px;")
        self.load_btn.clicked.connect(self.load_model)
        
        self.unload_btn = QPushButton("🗑️ Unload Model (Free VRAM)")
        self.unload_btn.setStyleSheet("background-color: #dc3545; color: white; font-weight: bold; padding: 8px;")
        self.unload_btn.clicked.connect(self.unload_model)
        self.unload_btn.setEnabled(False)
        
        btn_layout.addWidget(self.load_btn)
        btn_layout.addWidget(self.unload_btn)
        gpu_layout.addLayout(btn_layout)

        self.model_status = QLabel("Status: No model loaded")
        self.model_status.setStyleSheet("font-weight: bold; color: orange;")
        gpu_layout.addWidget(self.model_status)

        gpu_group.setLayout(gpu_layout)
        main_layout.addWidget(gpu_group)

        # --- RAG PANEL ---
        rag_layout = QHBoxLayout()
        self.pdf_btn = QPushButton("📁 Upload Medical PDF (RAG)")
        self.pdf_btn.clicked.connect(self.load_pdf)
        self.pdf_status = QLabel("RAG: No PDFs loaded")
        rag_layout.addWidget(self.pdf_btn)
        rag_layout.addWidget(self.pdf_status)
        rag_layout.addStretch()
        main_layout.addLayout(rag_layout)

        # --- CHAT INTERFACE ---
        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.chat_display.setFont(QFont("Consolas", 10))
        self.chat_display.setStyleSheet("background-color: #1e1e1e; color: #d4d4d4; padding: 10px;")
        main_layout.addWidget(self.chat_display)

        input_layout = QHBoxLayout()
        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("Ask a medical question...")
        self.input_field.returnPressed.connect(self.send_message)
        input_layout.addWidget(self.input_field)

        self.send_btn = QPushButton("Send")
        self.send_btn.clicked.connect(self.send_message)
        self.send_btn.setStyleSheet("background-color: #007acc; color: white; font-weight: bold; padding: 10px;")
        self.send_btn.setEnabled(False) # Disabled until model loads
        input_layout.addWidget(self.send_btn)

        main_layout.addLayout(input_layout)
        self.chat_display.append("<i>System: Please load a GGUF model to begin.</i><br>")

    def browse_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select GGUF Model", "", "GGUF Files (*.gguf)")
        if path:
            self.model_path_input.setText(path)

    def load_model(self):
        path = self.model_path_input.text().strip()
        if not os.path.exists(path):
            QMessageBox.warning(self, "Error", "Model file does not exist.")
            return

        self.load_btn.setEnabled(False)
        self.model_status.setText("Status: Allocating VRAM & Loading... (May take a few seconds)")
        self.model_status.setStyleSheet("font-weight: bold; color: blue;")
        QApplication.processEvents()

        try:
            # Pass the UI settings to llama.cpp
            self.llm = Llama(
                model_path=path,
                n_gpu_layers=self.gpu_layers_spin.value(),
                n_ctx=self.ctx_spin.value(),
                verbose=False
            )
            self.model_status.setText(f"Status: Model Loaded! ({os.path.basename(path)})")
            self.model_status.setStyleSheet("font-weight: bold; color: green;")
            self.send_btn.setEnabled(True)
            self.unload_btn.setEnabled(True)
            self.chat_display.append(f"<br><i>✅ Model successfully loaded into memory.</i><br>")
            
            # Reset chat history when changing models
            self.chat_history = []
            self.chat_display.clear()

        except Exception as e:
            self.llm = None
            self.model_status.setText("Status: FAILED TO LOAD (Out of VRAM?)")
            self.model_status.setStyleSheet("font-weight: bold; color: red;")
            QMessageBox.critical(self, "Loading Error", f"Failed to load model.\n\nIf you are out of VRAM, lower the 'GPU Layers' or 'Context Window' size.\n\nError: {str(e)}")
        
        self.load_btn.setEnabled(True)

    def unload_model(self):
        if self.is_generating:
            QMessageBox.warning(self, "Busy", "Cannot unload model while AI is generating text.")
            return

        if self.llm:
            del self.llm
            self.llm = None
            
            # Force Python to release the underlying C++ memory back to the OS/GPU driver
            gc.collect()
            
            self.model_status.setText("Status: Model Unloaded (VRAM Freed)")
            self.model_status.setStyleSheet("font-weight: bold; color: orange;")
            self.send_btn.setEnabled(False)
            self.unload_btn.setEnabled(False)
            self.chat_display.append("<br><i>🗑️ Model unloaded. GPU VRAM has been freed.</i><br>")

    def load_pdf(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Medical PDF", "", "PDF Files (*.pdf)")
        if file_path:
            try:
                count = rag_system.ingest_pdf(file_path)
                self.pdf_status.setText(f"RAG: Loaded {os.path.basename(file_path)} ({count} chunks)")
                self.chat_display.append(f"<i>📄 PDF '{os.path.basename(file_path)}' ingested.</i><br>")
            except Exception as e:
                self.chat_display.append(f"<i>❌ PDF Error: {str(e)}</i><br>")

    def send_message(self):
        if not self.llm:
            QMessageBox.warning(self, "Error", "Please load a model first.")
            return

        user_text = self.input_field.text().strip()
        if not user_text: return

        self.input_field.clear()
        self.chat_display.append(f"<b>You:</b> {user_text}<br>")
        self.chat_history.append({"role": "user", "content": user_text})

        # Lock UI Resources during generation
        self.is_generating = True
        self.send_btn.setEnabled(False)
        self.unload_btn.setEnabled(False) # PREVENT UNLOADING DURING GENERATION (Crash prevention)
        self.load_btn.setEnabled(False)

        # Pass the EXISTING model instance to the thread
        self.llm_thread = LLMThread(self.llm, self.chat_history)
        self.llm_thread.response_chunk.connect(self.display_response)
        self.llm_thread.tool_call_signal.connect(self.display_tool_status)
        self.llm_thread.finished_signal.connect(self.generation_done)
        self.llm_thread.start()

    def display_response(self, text: str):
        cursor = self.chat_display.textCursor()
        cursor.movePosition(cursor.End)
        cursor.insertText(text)
        self.chat_display.setTextCursor(cursor)
        self.chat_display.ensureCursorVisible()

    def display_tool_status(self, text: str):
        self.chat_display.append(f"<i>{text}</i>")

    def generation_done(self):
        self.chat_display.append("<br>")
        self.is_generating = False
        
        # Unlock UI Resources
        self.send_btn.setEnabled(True)
        self.unload_btn.setEnabled(True)
        self.load_btn.setEnabled(True)
        self.input_field.setFocus()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MedicalAssistantGUI()
    window.show()
    sys.exit(app.exec())