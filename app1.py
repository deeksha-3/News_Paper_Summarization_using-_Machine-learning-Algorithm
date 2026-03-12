import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinter.scrolledtext import ScrolledText
import nltk
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import numpy as np
from newspaper import Article
import PyPDF2
import docx

# Download tokenizer if first time
nltk.download('punkt')

# ----------------------------
# TextRank Machine Learning Summarizer
# ----------------------------
def summarize_text_ml(text, num_sentences=5):
    if len(text.strip()) == 0:
        messagebox.showwarning("Warning", "Please enter some text")
        return ""
    
    sentences = sent_tokenize(text)
    if len(sentences) <= num_sentences:
        return text
    
    # TF-IDF vectorization
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(sentences)
    
    # Compute sentence similarity matrix
    sim_matrix = cosine_similarity(tfidf_matrix)
    np.fill_diagonal(sim_matrix, 0)  # remove self-similarity
    
    # Build graph and apply PageRank
    nx_graph = nx.from_numpy_array(sim_matrix)
    scores = nx.pagerank(nx_graph)
    
    # Rank sentences and select top ones
    ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)
    
    # Keep top sentences in original order
    top_sentences = sorted([s for score, s in ranked_sentences[:num_sentences]], key=lambda x: sentences.index(x))
    return " ".join(top_sentences)

# ----------------------------
# Input Type Functions
# ----------------------------
def summarize_url(url_entry, input_box, output_box, progress):
    url = url_entry.get()
    if url.strip() == "":
        messagebox.showwarning("Error", "Enter URL")
        return
    try:
        article = Article(url)
        article.download()
        article.parse()
        text = article.text
        input_box.delete('1.0', tk.END)
        input_box.insert(tk.END, text)
        progress['value'] = 30
        root.update_idletasks()
        summary = summarize_text_ml(text)
        progress['value'] = 100
        output_box.delete('1.0', tk.END)
        output_box.insert(tk.END, summary)
    except Exception as e:
        messagebox.showerror("Error", f"Unable to fetch article\n{e}")

def summarize_text(input_box, output_box, progress):
    text = input_box.get('1.0', tk.END)
    progress['value'] = 20
    root.update_idletasks()
    summary = summarize_text_ml(text)
    progress['value'] = 100
    output_box.delete('1.0', tk.END)
    output_box.insert(tk.END, summary)

def upload_pdf(input_box, output_box, progress):
    file = filedialog.askopenfilename(filetypes=[("PDF files","*.pdf")])
    if file == "":
        return
    try:
        text = ""
        with open(file,"rb") as pdf:
            reader = PyPDF2.PdfReader(pdf)
            for page in reader.pages:
                text += page.extract_text() or ""
        input_box.delete('1.0', tk.END)
        input_box.insert(tk.END, text)
        progress['value'] = 20
        root.update_idletasks()
        summary = summarize_text_ml(text)
        progress['value'] = 100
        output_box.delete('1.0', tk.END)
        output_box.insert(tk.END, summary)
    except Exception as e:
        messagebox.showerror("Error", f"Cannot read PDF\n{e}")

def upload_docx(input_box, output_box, progress):
    file = filedialog.askopenfilename(filetypes=[("Word Files","*.docx")])
    if file == "":
        return
    try:
        text = ""
        doc = docx.Document(file)
        for para in doc.paragraphs:
            text += para.text + "\n"
        input_box.delete('1.0', tk.END)
        input_box.insert(tk.END, text)
        progress['value'] = 20
        root.update_idletasks()
        summary = summarize_text_ml(text)
        progress['value'] = 100
        output_box.delete('1.0', tk.END)
        output_box.insert(tk.END, summary)
    except Exception as e:
        messagebox.showerror("Error", f"Cannot read Word file\n{e}")

def clear_box(box):
    box.delete('1.0', tk.END)

def save_summary(output_box):
    text = output_box.get('1.0', tk.END)
    if text.strip() == "":
        messagebox.showwarning("Warning","No summary to save")
        return
    file = filedialog.asksaveasfilename(defaultextension=".txt",
                                        filetypes=[("Text Files","*.txt")])
    if file:
        with open(file,"w", encoding="utf-8") as f:
            f.write(text)
        messagebox.showinfo("Saved","Summary saved successfully!")

# ----------------------------
# GUI Setup
# ----------------------------
root = tk.Tk()
root.title("✨ AI Text Summarizer")
root.geometry("950x750")
root.configure(bg="#1e1e2f")

title = tk.Label(root, text="Machine Learning Text Summarizer",
                 font=("Helvetica",22,"bold"), bg="#1e1e2f", fg="#00f5d4")
title.pack(pady=10)

notebook = ttk.Notebook(root)
notebook.pack(expand=True, fill='both', padx=10, pady=10)

# --- URL Tab ---
tab_url = tk.Frame(notebook, bg="#2b2b3c")
notebook.add(tab_url, text="🌐 URL")

url_label = tk.Label(tab_url, text="Enter Article URL:", bg="#2b2b3c", fg="white", font=("Arial",12))
url_label.pack(pady=5)
url_entry = tk.Entry(tab_url, width=90, bg="#3a3a4d", fg="white", font=("Arial",11))
url_entry.pack(pady=5)

url_btn_frame = tk.Frame(tab_url, bg="#2b2b3c")
url_btn_frame.pack(pady=5)
progress_url = ttk.Progressbar(url_btn_frame, orient="horizontal", length=400, mode="determinate")
progress_url.grid(row=0, column=0, padx=5, pady=5)
input_url_text = ScrolledText(tab_url, width=100, height=12, bg="#3a3a4d", fg="white", insertbackground="white")
input_url_text.pack(pady=5)
output_url_text = ScrolledText(tab_url, width=100, height=10, bg="#2b2b3c", fg="#00f5d4", insertbackground="white")
output_url_text.pack(pady=5)

tk.Button(url_btn_frame, text="Summarize URL", bg="#ff6b6b", fg="white",
          command=lambda: summarize_url(url_entry,input_url_text,output_url_text,progress_url)).grid(row=0,column=1,padx=5)
tk.Button(url_btn_frame, text="Clear Input", bg="#ef476f", fg="white",
          command=lambda: clear_box(input_url_text)).grid(row=0,column=2,padx=5)
tk.Button(tab_url, text="Save Summary", bg="#06d6a0", fg="black",
          command=lambda: save_summary(output_url_text)).pack(pady=5)

# --- Text Tab ---
tab_text = tk.Frame(notebook, bg="#2b2b3c")
notebook.add(tab_text, text="📝 Paste Text")
input_text_box = ScrolledText(tab_text, width=100, height=14, bg="#3a3a4d", fg="white", insertbackground="white")
input_text_box.pack(pady=10)
output_text_box = ScrolledText(tab_text, width=100, height=10, bg="#2b2b3c", fg="#00f5d4", insertbackground="white")
output_text_box.pack(pady=5)
progress_text = ttk.Progressbar(tab_text, orient="horizontal", length=400, mode="determinate")
progress_text.pack(pady=5)
btn_frame_text = tk.Frame(tab_text, bg="#2b2b3c")
btn_frame_text.pack(pady=5)
tk.Button(btn_frame_text,text="Summarize",bg="#06d6a0", fg="black",
          command=lambda: summarize_text(input_text_box,output_text_box,progress_text)).grid(row=0,column=0,padx=5)
tk.Button(btn_frame_text,text="Clear Input",bg="#ef476f", fg="white",
          command=lambda: clear_box(input_text_box)).grid(row=0,column=1,padx=5)
tk.Button(btn_frame_text,text="Save Summary",bg="#ffd166", fg="black",
          command=lambda: save_summary(output_text_box)).grid(row=0,column=2,padx=5)

# --- PDF/Word Tab ---
tab_files = tk.Frame(notebook, bg="#2b2b3c")
notebook.add(tab_files, text="📄 PDF / Word")
file_btn_frame = tk.Frame(tab_files, bg="#2b2b3c")
file_btn_frame.pack(pady=10)
progress_file = ttk.Progressbar(file_btn_frame, orient="horizontal", length=400, mode="determinate")
progress_file.grid(row=0,column=0,padx=5)
tk.Button(file_btn_frame,text="Upload PDF",bg="#ffd166", fg="black",
          command=lambda: upload_pdf(input_file_text,output_file_text,progress_file)).grid(row=0,column=1,padx=5)
tk.Button(file_btn_frame,text="Upload Word",bg="#118ab2", fg="white",
          command=lambda: upload_docx(input_file_text,output_file_text,progress_file)).grid(row=0,column=2,padx=5)
tk.Button(file_btn_frame,text="Clear Input",bg="#ef476f", fg="white",
          command=lambda: clear_box(input_file_text)).grid(row=0,column=3,padx=5)

input_file_text = ScrolledText(tab_files, width=100, height=14, bg="#3a3a4d", fg="white", insertbackground="white")
input_file_text.pack(pady=10)
output_file_text = ScrolledText(tab_files, width=100, height=10, bg="#2b2b3c", fg="#00f5d4", insertbackground="white")
output_file_text.pack(pady=5)
tk.Button(tab_files, text="Save Summary", bg="#06d6a0", fg="black",
          command=lambda: save_summary(output_file_text)).pack(pady=5)

# ----------------------------
# Run App
# ----------------------------
root.mainloop()