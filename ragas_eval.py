"""
ragas_eval.py — CSE 635 Milestone 2 Baseline Evaluation
Judge: OpenAI GPT-4o-mini | Retriever: ChromaDB | Generator: Qwen2.5/Llama

pip install ragas langchain-openai datasets chromadb sentence-transformers pandas python-dotenv
"""

import os, json
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextRecall, ContextPrecision
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import chromadb
from dotenv import load_dotenv

# ── LOAD ENV ──────────────────────────────────────────────────────────────────
load_dotenv()
OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY")
CHROMA_PATH       = os.getenv("CHROMA_PATH", "./data/chroma")
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "ub_cse")
TOP_K             = int(os.getenv("TOP_K", "5"))
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# ── TEST SET (50 questions) ───────────────────────────────────────────────────
TEST_DATA = [
    # MS Program (10)
    ("MS Program", "How many credit hours are required for the MS CSE at UB?", "The MS CSE at UB requires 30 credit hours of graduate coursework."),
    ("MS Program", "What tracks are available in the MS CSE program at UB?", "The MS CSE program offers a thesis track and a course-only non-thesis track."),
    ("MS Program", "What is the minimum GPA to stay in good standing in MS CSE at UB?", "MS CSE students must maintain a minimum cumulative GPA of 3.0."),
    ("MS Program", "Can MS CSE students at UB specialize in artificial intelligence?", "Yes, the MS CSE program includes an AI and Machine Learning concentration."),
    ("MS Program", "How long does the MS CSE program typically take at UB?", "Full-time students typically complete the MS CSE program in 1.5 to 2 years."),
    ("MS Program", "Is a thesis required for the MS CSE at UB?", "No, the MS CSE offers both a thesis and a course-only track."),
    ("MS Program", "What concentrations exist in the MS CSE program at UB?", "Concentrations include AI and Machine Learning, Data-Intensive Computing, Security, and Systems."),
    ("MS Program", "Are there core required courses in the MS CSE program at UB?", "Yes, students must complete core courses in algorithms, systems, and theory."),
    ("MS Program", "Can MS CSE students take courses outside the CSE department at UB?", "Yes, a limited number of approved elective credits outside CSE are allowed with advisor approval."),
    ("MS Program", "What is the difference between thesis and non-thesis MS CSE at UB?", "The thesis track requires original research and a thesis worth 6 credits; the non-thesis track requires 30 coursework credits."),
    # PhD Program (6)
    ("PhD Program", "What doctoral degree does UB CSE offer?", "UB CSE offers a Doctor of Philosophy (PhD) in Computer Science and Engineering."),
    ("PhD Program", "Is a qualifying exam required for the CSE PhD at UB?", "Yes, PhD students must pass a qualifying examination."),
    ("PhD Program", "What does completing a PhD in CSE at UB require?", "It requires passing a qualifying exam, conducting original research, and defending a doctoral dissertation."),
    ("PhD Program", "Does UB CSE offer funding for PhD students?", "Yes, research and teaching assistantships with stipend and tuition coverage are available to qualified PhD students."),
    ("PhD Program", "Can a student with only a bachelor's degree apply to the UB CSE PhD program?", "Yes, students with a bachelor's degree can apply directly to the PhD program."),
    ("PhD Program", "What research areas can PhD students pursue at UB CSE?", "PhD students can pursue AI, machine learning, computer vision, NLP, cybersecurity, systems, and data science."),
    # Undergrad (5)
    ("Undergraduate", "What undergraduate degrees does UB CSE offer?", "UB CSE offers a BS in Computer Science and a BS in Computer Engineering."),
    ("Undergraduate", "Is there an accelerated BS/MS program at UB CSE?", "Yes, qualified undergraduates can complete both degrees in approximately five years."),
    ("Undergraduate", "What is the introductory CS course at UB CSE?", "CSE 115 Introduction to Computer Science is the foundational programming course for undergraduates."),
    ("Undergraduate", "Does UB CSE offer a minor in computer science?", "Yes, a minor in computer science is available to students from other majors."),
    ("Undergraduate", "What math is required for the BS Computer Science at UB?", "Required math includes calculus, linear algebra, discrete mathematics, and probability/statistics."),
    # Course Info (10)
    ("Course Info", "What is CSE 574 about?", "CSE 574 is Introduction to Machine Learning covering supervised learning, unsupervised learning, neural networks, and statistical learning theory."),
    ("Course Info", "What are the prerequisites for CSE 574?", "CSE 574 requires background in probability, linear algebra, and programming equivalent to CSE 474 or instructor consent."),
    ("Course Info", "What does CSE 535 cover?", "CSE 535 is Mobile Application Development covering design and implementation of iOS and Android apps."),
    ("Course Info", "What is CSE 562 about?", "CSE 562 is Database Systems covering relational models, SQL, query optimization, transaction management, and storage."),
    ("Course Info", "What does CSE 531 cover?", "CSE 531 is Algorithm Analysis and Design covering dynamic programming, greedy algorithms, divide and conquer, and complexity theory."),
    ("Course Info", "What is CSE 676 about?", "CSE 676 is Deep Learning covering CNN, RNN, and transformer architectures and their applications."),
    ("Course Info", "What is CSE 635 at UB?", "CSE 635 is NLP and Text Mining covering natural language processing, information extraction, text classification, and text mining."),
    ("Course Info", "How many credits is CSE 574?", "CSE 574 is a 3 credit hour graduate course."),
    ("Course Info", "What is CSE 589 about?", "CSE 589 is Modern Networking Concepts covering software-defined networking, network virtualization, and network security."),
    ("Course Info", "What language is used in CSE 116?", "CSE 116 Introduction to Computer Science II uses Python or Java to teach object-oriented programming."),
    # Faculty & Research (9)
    ("Faculty & Research", "What are Professor Rohini Srihari's research interests?", "Professor Srihari's interests include NLP, information extraction, multimodal data analysis, and AI for social impact."),
    ("Faculty & Research", "What research areas does UB CSE focus on?", "UB CSE research covers AI, machine learning, data mining, computer vision, NLP, cybersecurity, systems, and networking."),
    ("Faculty & Research", "Does UB CSE have an AI research group?", "Yes, UB CSE has an AI, Machine Learning, and Data Mining research group with multiple faculty members."),
    ("Faculty & Research", "How many faculty are in UB CSE?", "The UB CSE department has over 40 full-time faculty members."),
    ("Faculty & Research", "Does UB CSE have research labs students can join?", "Yes, UB CSE has research labs in AI, systems, security, and data science that graduate students can join."),
    ("Faculty & Research", "What does the AI and ML research group at UB CSE focus on?", "The group focuses on deep learning, NLP, computer vision, knowledge representation, and data-driven decision making."),
    ("Faculty & Research", "Can MS students do research at UB CSE?", "Yes, especially thesis-track MS students can join faculty research labs."),
    ("Faculty & Research", "What security research is done at UB CSE?", "Security research at UB CSE covers network security, system security, cryptography, and privacy-preserving computing."),
    ("Faculty & Research", "Does UB CSE collaborate with industry?", "Yes, UB CSE collaborates with industry partners through grants and externally funded research projects."),
    # Admissions (5)
    ("Admissions", "What are the MS CSE application requirements at UB?", "Requirements include a bachelor's in CS or related field, transcripts, recommendation letters, a statement of purpose, and GRE scores (varies by cycle)."),
    ("Admissions", "Is the GRE required for MS CSE at UB?", "GRE requirements vary by application cycle; check the current UB CSE admissions page for up-to-date requirements."),
    ("Admissions", "What is the MS CSE application deadline at UB?", "Deadlines vary by semester; refer to the official UB CSE graduate admissions page for current dates."),
    ("Admissions", "Can international students apply to MS CSE at UB?", "Yes, international students must also submit TOEFL or IELTS scores for English proficiency."),
    ("Admissions", "What background is recommended for MS CSE applicants?", "A background in computer science or engineering with coursework in programming, algorithms, and mathematics is recommended."),
    # Out-of-Scope / Guardrail (5)
    ("Out-of-Scope", "What is the best pizza place near UB?", "This is outside the scope of the UB CSE chatbot. I can only answer CSE department related questions."),
    ("Out-of-Scope", "Write me a Python script to reverse a string.", "This is outside the scope of the UB CSE chatbot. I can only answer CSE department related questions."),
    ("Out-of-Scope", "What courses does the UB Medical School offer?", "This is outside the scope of the UB CSE chatbot. I can only answer CSE department related questions."),
    ("Out-of-Scope", "Who won the NBA championship last year?", "This is outside the scope of the UB CSE chatbot. I can only answer CSE department related questions."),
    ("Out-of-Scope", "What is the weather like in Buffalo in winter?", "This is outside the scope of the UB CSE chatbot. I can only answer CSE department related questions."),
]

# ── RETRIEVER ─────────────────────────────────────────────────────────────────
from chromadb.config import Settings
from langchain_ollama import OllamaEmbeddings

client     = chromadb.PersistentClient(path=CHROMA_PATH,
                                       settings=Settings(anonymized_telemetry=False))
collection = client.get_collection(CHROMA_COLLECTION)
embedder   = OllamaEmbeddings(model="nomic-embed-text")

def retrieve(question):
    vec = embedder.embed_query(question)
    res = collection.query(query_embeddings=[vec],
                           n_results=TOP_K, include=["documents"])
    return res["documents"][0]

# ── YOUR LLAMA GENERATOR — replace body with your actual call ─────────────────
def generate(question, contexts):
    context_str = "\n\n".join(contexts)
    prompt = f"""You are a UB CSE department assistant. You MUST answer using the context below.
Always give a specific, complete answer of 2-3 sentences using facts from the context.
Only say you don't know if the topic is completely absent from the context.

Context:
{context_str}

Question: {question}
Answer (use facts from the context above):"""
    import requests
    response = requests.post("http://localhost:11434/api/generate",
                             json={"model": "llama3.2:3b", "prompt": prompt, "stream": False})
    return response.json()["response"].strip()

# ── RUN PIPELINE ──────────────────────────────────────────────────────────────
print(f"Running pipeline on {len(TEST_DATA)} questions...")
rows = []
for cat, q, gt in TEST_DATA:
    chunks = retrieve(q)
    try:
        answer = generate(q, chunks)
    except NotImplementedError:
        answer = "PLACEHOLDER"
    rows.append({"category": cat, "question": q, "answer": answer,
                 "contexts": chunks, "ground_truth": gt})
    print(f"  ✓ {q[:60]}...")

# ── RAGAS ─────────────────────────────────────────────────────────────────────
print("\nRunning RAGAS evaluation...")
llm   = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=OPENAI_API_KEY)
emb   = OpenAIEmbeddings(model="text-embedding-3-small", api_key=OPENAI_API_KEY)
ds    = Dataset.from_list([{k: v for k, v in r.items()} for r in rows])

scores = evaluate(ds, metrics=[Faithfulness(), AnswerRelevancy(),
                                ContextRecall(), ContextPrecision()],
                  llm=llm, embeddings=emb, raise_exceptions=False)

# ── RESULTS ───────────────────────────────────────────────────────────────────
METRICS = ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]
METRIC_KEYS = METRICS

print("\n" + "="*50)
for m in METRICS:
    print(f"  {m:<26}: {float(scores[m]) if not isinstance(scores[m], list) else float(scores[m][0]):.4f}")
print("="*50)

# Save outputs
df = scores.to_pandas()
df["category"] = [r["category"] for r in rows]
df.to_csv("ragas_results.csv", index=False)

summary = {m: round(float(scores[m]), 4) for m in METRICS}
json.dump(summary, open("ragas_summary.json", "w"), indent=2)

cat_scores = df.groupby("category")[METRICS].mean().round(4)
cat_scores.to_csv("ragas_category_scores.csv")

print("\n✓ ragas_results.csv")
print("✓ ragas_summary.json  <- paste into paper")
print("✓ ragas_category_scores.csv <- table for paper")
print("\nCategory breakdown:")
print(cat_scores.to_string())