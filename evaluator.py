"""
evaluator.py — Evaluation Suite for UB CSE Chatbot
====================================================
Runs four evaluation tracks as required by the rubric:

  1. RAGAS Faithfulness & Answer Relevance
     - Faithfulness:      is the answer grounded in the retrieved context?
     - Answer Relevance:  does the answer address the question asked?
     - Context Recall:    did retrieval surface the relevant chunks?

  2. Retrieval Precision (Hit Rate @ k)
     - For each eval query, checks if the correct document is in top-k
     - Reports Recall@1, Recall@3, Recall@5, Recall@10
     - Also reports MRR (Mean Reciprocal Rank) and nDCG@5

  3. Latency Benchmarking
     - Measures Time to First Token (TTFT) for each query
     - Reports p50, p90, p95 latency
     - Target: TTFT < 2000ms

  4. Robustness / Guardrail Testing
     - Runs the built-in guardrail test suite
     - Reports pass rate on out-of-scope and adversarial queries

Output:
  - data/eval/eval_results.json   — full results
  - data/eval/eval_summary.json   — summary metrics for the report
  - Printed table to stdout

Built-in eval dataset (no external file needed):
  50 curated queries covering courses, faculty, programs, admissions,
  research areas — with expected answer keywords for hit-rate testing.

HOW TO RUN:
  # Full evaluation (requires indexes + Ollama running):
  python evaluator.py

  # Just retrieval metrics (no Ollama needed):
  python evaluator.py --retrieval-only

  # Just latency (requires Ollama):
  python evaluator.py --latency-only

  # Just guardrails (no indexes needed):
  python evaluator.py --guardrails-only

  # Save results:
  python evaluator.py --output data/eval/my_results.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from config import cfg
from guardrails import run_test_suite
from retriever import get_retriever
from utils import get_logger, save_json

log = get_logger(__name__)


# ── Built-in evaluation dataset ───────────────────────────────────────────────
# Each entry: {query, expected_keywords, relevant_page_types, category}
# expected_keywords: list of strings that SHOULD appear in top-k results
# relevant_page_types: page types that count as "relevant" hits

EVAL_DATASET = [{'query': 'What are the prerequisites for CSE 574?',
  'category': 'course_prereq',
  'expected_answer': 'CSE 574 is Introduction to Machine Learning. The catalog lists it as a '
                     '3-credit course. Use the official catalog/source chunk to verify '
                     'prerequisites because prerequisite wording can vary by catalog year.',
  'expected_keywords': ['CSE 574', 'Introduction to Machine Learning', 'Credits: 3'],
  'expected_answer_kw': ['CSE 574', 'Introduction to Machine Learning', '3'],
  'relevant_page_types': ['courses', 'catalog_course', 'catalog_program'],
  'source_url': 'https://catalogs.buffalo.edu/preview_course_nopop.php?catoid=2&coid=6166'},
 {'query': 'What is CSE 573 about?',
  'category': 'course_description',
  'expected_answer': 'CSE 573 is Introduction to Computer Vision and Image Processing. It '
                     'introduces areas of AI dealing with fundamental issues and techniques of '
                     'computer vision and image processing.',
  'expected_keywords': ['CSE 573',
                        'computer vision',
                        'image processing',
                        'artificial intelligence'],
  'expected_answer_kw': ['computer vision', 'image processing'],
  'relevant_page_types': ['courses', 'catalog_course'],
  'source_url': 'https://catalogs.buffalo.edu/preview_course_nopop.php?catoid=2&coid=6257'},
 {'query': 'How many credits is CSE 531?',
  'category': 'course_credits',
  'expected_answer': 'CSE 531, Analysis of Algorithms I, is listed as a 3-credit course.',
  'expected_keywords': ['CSE 531', 'Analysis of Algorithms I', 'Credits: 3'],
  'expected_answer_kw': ['3', 'credit'],
  'relevant_page_types': ['courses', 'catalog_course'],
  'source_url': 'https://catalogs.buffalo.edu/preview_course_nopop.php?catoid=2&coid=6239'},
 {'query': 'What is CSE 442 about?',
  'category': 'course_description',
  'expected_answer': 'CSE 442 is Software Engineering. Topics include software life-cycle models, '
                     'architectural and design approaches, systematic software testing, coding, '
                     'and documentation.',
  'expected_keywords': ['CSE 442', 'Software Engineering', 'software life-cycle', 'testing'],
  'expected_answer_kw': ['software', 'life-cycle', 'testing'],
  'relevant_page_types': ['courses', 'catalog_course'],
  'source_url': 'https://catalogs.buffalo.edu/preview_course_nopop.php?catoid=11&coid=70960'},
 {'query': 'Describe CSE 116',
  'category': 'course_description',
  'expected_answer': 'CSE 116 is Introduction to Computer Science II.',
  'expected_keywords': ['CSE 116', 'Introduction to Computer Science II'],
  'expected_answer_kw': ['Introduction to Computer Science II'],
  'relevant_page_types': ['courses', 'catalog_course'],
  'source_url': 'https://engineering.buffalo.edu/computer-science-engineering/undergraduate/courses/class-schedule.html'},
 {'query': 'What are the MS CSE degree requirements?',
  'category': 'program_requirements',
  'expected_answer': 'The CSE MS program requires 30 credit hours. The curriculum includes four '
                     'breadth computer science courses, three focus courses that may form a '
                     'specialization, three electives, and a capstone course.',
  'expected_keywords': ['MS', '30 credit hours', 'breadth', 'focus', 'electives', 'capstone'],
  'expected_answer_kw': ['30', 'breadth', 'focus', 'electives', 'capstone'],
  'relevant_page_types': ['degree_requirements', 'catalog_program'],
  'source_url': 'https://engineering.buffalo.edu/computer-science-engineering/graduate/degrees-and-programs/ms-in-computer-science-and-engineering/ms-specializations.html'},
 {'query': 'How many credits for the MS in CSE?',
  'category': 'program_credits',
  'expected_answer': 'The Computer Science and Engineering MS requires 30 credit hours.',
  'expected_keywords': ['MS', '30 credit hours'],
  'expected_answer_kw': ['30', 'credit'],
  'relevant_page_types': ['degree_requirements', 'catalog_program'],
  'source_url': 'https://engineering.buffalo.edu/computer-science-engineering/graduate/degrees-and-programs/ms-in-computer-science-and-engineering/ms-specializations.html'},
 {'query': 'What are the PhD program requirements?',
  'category': 'program_requirements',
  'expected_answer': 'UB CSE states that PhD degree requirements are defined by the CSE Graduate '
                     "Handbook in force during the student's matriculation year.",
  'expected_keywords': ['PhD',
                        'degree requirements',
                        'CSE Graduate Handbook',
                        'matriculation year'],
  'expected_answer_kw': ['Graduate Handbook', 'matriculation'],
  'relevant_page_types': ['degree_requirements', 'catalog_program', 'handbook_pdf'],
  'source_url': 'https://engineering.buffalo.edu/computer-science-engineering/graduate/degrees-and-programs/phd-in-computer-science-and-engineering.html'},
 {'query': 'What GPA is required for MS CSE admission?',
  'category': 'admissions',
  'expected_answer': 'UB CSE says there is no strict minimum GPA requirement, but applicants are '
                     'expected to have a GPA higher than 3.0 on a 4.0 scale. The average GPA of '
                     'admitted applicants is around 3.4.',
  'expected_keywords': ['GPA', '3.0', '3.4', 'admission'],
  'expected_answer_kw': ['3.0', '3.4', 'GPA'],
  'relevant_page_types': ['admissions'],
  'source_url': 'https://engineering.buffalo.edu/computer-science-engineering/graduate/admissions/faq.html'},
 {'query': 'Is GRE required for MS CSE admission?',
  'category': 'admissions',
  'expected_answer': 'The GRE is not required for MS or PhD applicants.',
  'expected_keywords': ['GRE', 'not required', 'MS', 'PhD'],
  'expected_answer_kw': ['GRE', 'not required'],
  'relevant_page_types': ['admissions'],
  'source_url': 'https://engineering.buffalo.edu/computer-science-engineering/graduate/admissions/application-materials.html'},
 {'query': 'What documents are needed to apply to CSE PhD?',
  'category': 'admissions',
  'expected_answer': 'The application asks for personal and biographical information, contact '
                     'information, citizenship details, college/university and degree details, '
                     'recommender information, and supporting materials. Three letters of '
                     'recommendation are required for PhD applicants.',
  'expected_keywords': ['application',
                        'supporting materials',
                        'three letters of recommendation',
                        'PhD'],
  'expected_answer_kw': ['three', 'letters', 'recommendation'],
  'relevant_page_types': ['admissions'],
  'source_url': 'https://engineering.buffalo.edu/computer-science-engineering/graduate/admissions/application-materials.html'},
 {'query': 'What is the application deadline for MS CSE?',
  'category': 'admissions',
  'expected_answer': 'For fall admission, students are encouraged to apply by December 10 for '
                     'early scholarship consideration and by December 31 for full funding or '
                     'fellowship consideration. Spring deadlines are September 30 for '
                     'international applicants and October 31 for domestic applicants.',
  'expected_keywords': ['deadline', 'December 10', 'December 31', 'September 30', 'October 31'],
  'expected_answer_kw': ['December', 'September', 'October'],
  'relevant_page_types': ['admissions'],
  'source_url': 'https://engineering.buffalo.edu/computer-science-engineering/graduate/admissions/application-materials.html'},
 {'query': "What are Junsong Yuan's research areas?",
  'category': 'faculty_research',
  'expected_answer': "Junsong Yuan's listed research topics include computer vision, pattern "
                     'recognition, video analytics, and large-scale visual search and mining.',
  'expected_keywords': ['Junsong Yuan',
                        'computer vision',
                        'pattern recognition',
                        'video analytics'],
  'expected_answer_kw': ['computer vision', 'pattern recognition', 'video analytics'],
  'relevant_page_types': ['faculty_profile', 'research'],
  'source_url': 'https://engineering.buffalo.edu/computer-science-engineering/graduate/academic-advisement.host.html/content/shared/engineering/computer-science-engineering/profiles/faculty/ladder/yuan-junsong.detail.students.html'},
 {'query': "What is Junsong Yuan's email?",
  'category': 'faculty_contact',
  'expected_answer': "Junsong Yuan's UB CSE email is jsyuan@buffalo.edu.",
  'expected_keywords': ['Junsong Yuan', 'jsyuan@buffalo.edu'],
  'expected_answer_kw': ['jsyuan@buffalo.edu'],
  'relevant_page_types': ['faculty_profile'],
  'source_url': 'https://engineering.buffalo.edu/computer-science-engineering/graduate/academic-advisement.html'},
 {'query': 'Who are the NLP faculty at UB CSE?',
  'category': 'faculty_research',
  'expected_answer': 'The UB CSE Natural Language Processing research area page lists affiliated '
                     'faculty including Changyou Chen, Kenneth Joseph, and Rohini Srihari.',
  'expected_keywords': ['Natural Language Processing',
                        'Changyou Chen',
                        'Kenneth Joseph',
                        'Rohini Srihari'],
  'expected_answer_kw': ['Changyou Chen', 'Kenneth Joseph', 'Rohini Srihari'],
  'relevant_page_types': ['faculty_profile', 'research'],
  'source_url': 'https://engineering.buffalo.edu/computer-science-engineering/research/research-areas/artificial-intelligence/natural-language-processing.html'},
 {'query': 'Which faculty work on machine learning?',
  'category': 'faculty_research',
  'expected_answer': 'The AI, Machine Learning and Data Mining research area page lists affiliated '
                     'faculty such as Roshan Ayyalasomayajula, Varun Chandola, Changyou Chen, '
                     'Sreyasee Das Bhattacharjee, and David Doermann.',
  'expected_keywords': ['machine learning',
                        'Roshan Ayyalasomayajula',
                        'Varun Chandola',
                        'Changyou Chen'],
  'expected_answer_kw': ['machine learning', 'faculty'],
  'relevant_page_types': ['faculty_profile', 'research'],
  'source_url': 'https://engineering.buffalo.edu/computer-science-engineering/research/research-areas/artificial-intelligence/artificial-intelligence-and-machine-learning-and-data-mining.html'},
 {'query': 'Who teaches computer vision at UB CSE?',
  'category': 'faculty_course',
  'expected_answer': 'The Computer Vision research area page lists affiliated faculty including '
                     'Changyou Chen, Sreyasee Das Bhattacharjee, David Doermann, and Mingchen Gao.',
  'expected_keywords': ['computer vision',
                        'Changyou Chen',
                        'Sreyasee Das Bhattacharjee',
                        'David Doermann',
                        'Mingchen Gao'],
  'expected_answer_kw': ['computer vision', 'faculty'],
  'relevant_page_types': ['faculty_profile', 'research', 'courses'],
  'source_url': 'https://engineering.buffalo.edu/computer-science-engineering/research/research-areas/artificial-intelligence/computer-vision.html'},
 {'query': 'What research labs exist in UB CSE?',
  'category': 'research',
  'expected_answer': 'UB CSE has research centers, institutes, and groups such as CARA, CUBS, '
                     'CEDAR, CEISARE, Computing for Social Good, and departmental research labs '
                     'and groups.',
  'expected_keywords': ['research',
                        'labs',
                        'centers',
                        'groups',
                        'CARA',
                        'CUBS',
                        'CEDAR',
                        'CEISARE'],
  'expected_answer_kw': ['research', 'labs', 'centers'],
  'relevant_page_types': ['research'],
  'source_url': 'https://engineering.buffalo.edu/computer-science-engineering/research/research-centers-institutes-labs-and-groups.html'},
 {'query': 'What are the research areas in UB CSE?',
  'category': 'research',
  'expected_answer': 'UB CSE has a research areas page listing areas including artificial '
                     'intelligence, systems, and theory.',
  'expected_keywords': ['research areas', 'artificial intelligence', 'systems', 'theory'],
  'expected_answer_kw': ['research areas'],
  'relevant_page_types': ['research'],
  'source_url': 'https://engineering.buffalo.edu/computer-science-engineering/research/research-areas.html'},
 {'query': 'Are CSE PhD students fully funded?',
  'category': 'phd_funding',
  'expected_answer': 'UB CSE states that all CSE PhD students are fully funded through teaching '
                     'assistantships, research assistantships, and fellowships.',
  'expected_keywords': ['PhD',
                        'fully funded',
                        'Teaching Assistantships',
                        'Research Assistantships',
                        'Fellowships'],
  'expected_answer_kw': ['fully funded',
                         'Teaching Assistantships',
                         'Research Assistantships',
                         'Fellowships'],
  'relevant_page_types': ['admissions', 'graduate_programs'],
  'source_url': 'https://engineering.buffalo.edu/computer-science-engineering/graduate/admissions/faq.html'},
 {'query': 'Can I apply directly to the CSE PhD program without an MS degree?',
  'category': 'phd_admissions',
  'expected_answer': "A master's degree is not required for admission to the CSE PhD program, but "
                     'it is highly recommended.',
  'expected_keywords': ["master's degree", 'not required', 'PhD', 'highly recommended'],
  'expected_answer_kw': ['not required', 'highly recommended'],
  'relevant_page_types': ['admissions'],
  'source_url': 'https://engineering.buffalo.edu/computer-science-engineering/graduate/admissions/faq.html'},
 {'query': 'How many recommendation letters are required for the CSE PhD application?',
  'category': 'phd_admissions',
  'expected_answer': 'Three letters of recommendation are required to apply to the PhD program.',
  'expected_keywords': ['Three', 'letters of recommendation', 'PhD'],
  'expected_answer_kw': ['three', 'letters', 'recommendation'],
  'relevant_page_types': ['admissions'],
  'source_url': 'https://engineering.buffalo.edu/computer-science-engineering/graduate/admissions/application-materials.html'},
 {'query': 'How many recommendation letters are required for the MS application?',
  'category': 'ms_admissions',
  'expected_answer': 'Two letters of recommendation are required to apply to the MS program.',
  'expected_keywords': ['Two', 'letters of recommendation', 'MS'],
  'expected_answer_kw': ['two', 'letters', 'recommendation'],
  'relevant_page_types': ['admissions'],
  'source_url': 'https://engineering.buffalo.edu/computer-science-engineering/graduate/admissions/application-materials.html'},
 {'query': 'What is the application fee for CSE graduate programs?',
  'category': 'admissions',
  'expected_answer': 'A non-refundable application fee of $100 is required for Spring 2025 or '
                     'later terms.',
  'expected_keywords': ['application fee', '$100', 'Spring 2025'],
  'expected_answer_kw': ['$100', 'application fee'],
  'relevant_page_types': ['admissions'],
  'source_url': 'https://engineering.buffalo.edu/computer-science-engineering/graduate/admissions/application-materials.html'},
 {'query': 'What TOEFL iBT score does UB require for international applicants?',
  'category': 'international_admissions',
  'expected_answer': 'The UB minimum TOEFL iBT score listed for international applicants is 79.',
  'expected_keywords': ['TOEFL', 'IBT', '79', 'English Proficiency'],
  'expected_answer_kw': ['TOEFL', '79'],
  'relevant_page_types': ['admissions'],
  'source_url': 'https://engineering.buffalo.edu/computer-science-engineering/graduate/admissions/application-materials.html'},
 {'query': 'Can MS students build their own specialization?',
  'category': 'ms_specialization',
  'expected_answer': 'UB CSE says MS students may complete one or more specializations or build '
                     'their own specialization by selecting courses that align with their '
                     'interests.',
  'expected_keywords': ['specialization', 'build your own', 'courses', 'interests'],
  'expected_answer_kw': ['specialization', 'courses', 'interests'],
  'relevant_page_types': ['degree_requirements', 'catalog_program'],
  'source_url': 'https://engineering.buffalo.edu/computer-science-engineering/graduate/admissions/faq.html'},
 {'query': 'What courses make up the MS CSE curriculum structure?',
  'category': 'program_requirements',
  'expected_answer': 'The MS curriculum includes four breadth computer science courses, three '
                     'focus courses, three electives, and a capstone course.',
  'expected_keywords': ['four breadth', 'three focus', 'three electives', 'capstone'],
  'expected_answer_kw': ['breadth', 'focus', 'electives', 'capstone'],
  'relevant_page_types': ['degree_requirements', 'catalog_program'],
  'source_url': 'https://engineering.buffalo.edu/computer-science-engineering/graduate/degrees-and-programs/ms-in-computer-science-and-engineering/ms-specializations.html'},
 {'query': 'What is the AI/ML MS track capstone course?',
  'category': 'ms_track',
  'expected_answer': 'The AI/ML track page states that students complete a capstone class, CSE '
                     '573.',
  'expected_keywords': ['AI/ML', 'capstone', 'CSE 573'],
  'expected_answer_kw': ['CSE 573', 'capstone'],
  'relevant_page_types': ['degree_requirements', 'catalog_program', 'courses'],
  'source_url': 'https://engineering.buffalo.edu/computer-science-engineering/graduate/degrees-and-programs/ms-in-computer-science-and-engineering/ms-tracks.html'},
 {'query': 'What is the Systems MS track capstone course?',
  'category': 'ms_track',
  'expected_answer': 'The Systems track page states that students complete a capstone class, CSE '
                     '562.',
  'expected_keywords': ['Systems', 'capstone', 'CSE 562'],
  'expected_answer_kw': ['CSE 562', 'capstone'],
  'relevant_page_types': ['degree_requirements', 'catalog_program', 'courses'],
  'source_url': 'https://engineering.buffalo.edu/computer-science-engineering/graduate/degrees-and-programs/ms-in-computer-science-and-engineering/ms-tracks.html'},
 {'query': 'Which research centers are listed by UB CSE?',
  'category': 'research_labs',
  'expected_answer': 'The research centers, institutes, labs and groups page lists centers such as '
                     'CARA, CUBS, CEDAR, and CEISARE.',
  'expected_keywords': ['CARA', 'CUBS', 'CEDAR', 'CEISARE', 'Centers'],
  'expected_answer_kw': ['CARA', 'CUBS', 'CEDAR', 'CEISARE'],
  'relevant_page_types': ['research'],
  'source_url': 'https://engineering.buffalo.edu/computer-science-engineering/research/research-centers-institutes-labs-and-groups.html'},
 {'query': 'What does CEISARE research focus on?',
  'category': 'research_labs',
  'expected_answer': 'CEISARE focuses on computer security and information assurance, including '
                     'e-commerce, security, networks, secure voting, cyber-attack recognition, '
                     'insider threats, intrusion detection, and related topics.',
  'expected_keywords': ['CEISARE',
                        'computer security',
                        'information assurance',
                        'networks',
                        'secure voting'],
  'expected_answer_kw': ['computer security', 'information assurance'],
  'relevant_page_types': ['research', 'faculty_profile'],
  'source_url': 'https://engineering.buffalo.edu/computer-science-engineering/research/research-centers-institutes-labs-and-groups/center-of-excellence-in-information-systems-assurance-and-research-in-education.html'},
 {'query': 'What does CARA research focus on?',
  'category': 'research_labs',
  'expected_answer': 'CARA was founded to promote basic and applied big-data-related research and '
                     'multidisciplinary data analytics applications.',
  'expected_keywords': ['CARA', 'big data', 'data analytics', 'multidisciplinary'],
  'expected_answer_kw': ['big data', 'data analytics'],
  'relevant_page_types': ['research'],
  'source_url': 'https://engineering.buffalo.edu/computer-science-engineering/research/research-centers-institutes-labs-and-groups/center-for-analytics-research-and-applications.html'},
 {'query': 'Which faculty are listed on the Natural Language Processing research area page?',
  'category': 'graph_research_faculty',
  'expected_answer': 'The NLP research area page lists affiliated faculty including Changyou Chen, '
                     'Kenneth Joseph, and Rohini Srihari.',
  'expected_keywords': ['Natural Language Processing',
                        'Changyou Chen',
                        'Kenneth Joseph',
                        'Rohini Srihari'],
  'expected_answer_kw': ['Changyou Chen', 'Kenneth Joseph', 'Rohini Srihari'],
  'relevant_page_types': ['research', 'faculty_profile'],
  'source_url': 'https://engineering.buffalo.edu/computer-science-engineering/research/research-areas/artificial-intelligence/natural-language-processing.html'},
 {'query': 'Which faculty are listed on the Computer Vision research area page?',
  'category': 'graph_research_faculty',
  'expected_answer': 'The Computer Vision research area page lists affiliated faculty including '
                     'Changyou Chen, Sreyasee Das Bhattacharjee, David Doermann, and Mingchen Gao.',
  'expected_keywords': ['Computer Vision',
                        'Changyou Chen',
                        'Sreyasee Das Bhattacharjee',
                        'David Doermann',
                        'Mingchen Gao'],
  'expected_answer_kw': ['Computer Vision', 'faculty'],
  'relevant_page_types': ['research', 'faculty_profile'],
  'source_url': 'https://engineering.buffalo.edu/computer-science-engineering/research/research-areas/artificial-intelligence/computer-vision.html'},
 {'query': 'Which summer 2026 sections are listed for CSE 116?',
  'category': 'course_schedule',
  'expected_answer': 'The class schedule lists CSE 116 sections such as A, A1, B, and B1 for '
                     'summer 2026.',
  'expected_keywords': ['CSE 116', 'A', 'A1', 'B', 'B1', '05/26/2026'],
  'expected_answer_kw': ['CSE 116', 'A', 'A1', 'B', 'B1'],
  'relevant_page_types': ['courses', 'schedule_pdf', 'registrar'],
  'source_url': 'https://engineering.buffalo.edu/computer-science-engineering/undergraduate/courses/class-schedule.html'}]


# ── IR metrics ────────────────────────────────────────────────────────────────

def _hit_at_k(results: list[dict], keywords: list[str], k: int) -> bool:
    """Return True if any of the top-k results contain at least one keyword."""
    for r in results[:k]:
        text = (r.get("text") or "").lower()
        if any(kw.lower() in text for kw in keywords):
            return True
    return False


def _reciprocal_rank(results: list[dict], keywords: list[str]) -> float:
    """Return 1/rank of the first result containing a keyword, or 0."""
    for i, r in enumerate(results, 1):
        text = (r.get("text") or "").lower()
        if any(kw.lower() in text for kw in keywords):
            return 1.0 / i
    return 0.0


def _ndcg_at_k(results: list[dict], keywords: list[str], k: int) -> float:
    """Compute nDCG@k treating keyword-match as binary relevance."""
    def rel(r):
        text = (r.get("text") or "").lower()
        return 1 if any(kw.lower() in text for kw in keywords) else 0

    dcg  = sum(rel(results[i]) / math.log2(i + 2) for i in range(min(k, len(results))))
    # Ideal DCG: all relevant docs at top
    n_rel = sum(rel(r) for r in results[:k])
    idcg  = sum(1.0 / math.log2(i + 2) for i in range(n_rel))
    return dcg / idcg if idcg > 0 else 0.0


# ── Retrieval evaluation ──────────────────────────────────────────────────────

async def run_retrieval_eval(
    dataset: list[dict],
    k_values: list[int] = None,
    verbose: bool = True,
) -> dict:
    """
    Run IR evaluation: Recall@k, MRR, nDCG@5 across all eval queries.
    Does NOT require Ollama — only the indexes.
    """
    k_values = k_values or cfg.eval.k_values
    retriever = get_retriever()

    recall    = {k: [] for k in k_values}
    mrr_scores = []
    ndcg_scores = []
    latencies   = []

    print(f"\n{'='*65}")
    print(f"  RETRIEVAL EVALUATION  ({len(dataset)} queries)")
    print(f"{'='*65}")

    for i, item in enumerate(dataset, 1):
        query    = item["query"]
        keywords = item["expected_keywords"]

        t0 = time.perf_counter()
        try:
            results = retriever.retrieve_sync(query, top_k=max(k_values))
        except Exception as e:
            log.warning("Retrieval failed for %r: %s", query, e)
            results = []
        elapsed = (time.perf_counter() - t0) * 1000
        latencies.append(elapsed)

        # Compute metrics
        for k in k_values:
            recall[k].append(1.0 if _hit_at_k(results, keywords, k) else 0.0)

        rr   = _reciprocal_rank(results, keywords)
        ndcg = _ndcg_at_k(results, keywords, 5)
        mrr_scores.append(rr)
        ndcg_scores.append(ndcg)

        hit1 = "✅" if recall[1][-1] else "❌"
        if verbose:
            print(f"  {i:2}. {hit1} [{item['category']:22}] "
                  f"MRR={rr:.2f} nDCG@5={ndcg:.2f} "
                  f"({elapsed:.0f}ms)  {query[:50]}")

    # Aggregate
    results_out = {
        "recall_at_k": {k: round(sum(v)/len(v), 4) for k, v in recall.items()},
        "mrr":         round(sum(mrr_scores) / len(mrr_scores), 4),
        "ndcg_at_5":   round(sum(ndcg_scores) / len(ndcg_scores), 4),
        "latency": {
            "p50": round(sorted(latencies)[len(latencies)//2], 1),
            "p90": round(sorted(latencies)[int(len(latencies)*0.9)], 1),
            "p95": round(sorted(latencies)[int(len(latencies)*0.95)], 1),
            "mean": round(sum(latencies)/len(latencies), 1),
        },
        "n_queries": len(dataset),
    }

    print(f"\n  Results:")
    for k, v in results_out["recall_at_k"].items():
        print(f"    Recall@{k:<3} = {v:.4f}  ({v*100:.1f}%)")
    print(f"    MRR        = {results_out['mrr']:.4f}")
    print(f"    nDCG@5     = {results_out['ndcg_at_5']:.4f}")
    print(f"    Latency p50= {results_out['latency']['p50']:.0f}ms  "
          f"p90={results_out['latency']['p90']:.0f}ms  "
          f"p95={results_out['latency']['p95']:.0f}ms")

    return results_out


# ── RAGAS evaluation ──────────────────────────────────────────────────────────

async def run_ragas_eval(
    dataset: list[dict],
    n_samples: int = 10,
    verbose: bool = True,
) -> dict:
    """
    Run RAGAS faithfulness + answer relevance evaluation.
    Requires Ollama running with the configured model.
    Uses a subset of the dataset (n_samples) to keep runtime reasonable.
    """
    try:
        from ragas import evaluate as ragas_evaluate          # type: ignore
        from ragas.metrics import faithfulness, answer_relevancy, context_recall  # type: ignore
        from datasets import Dataset                           # type: ignore
    except ImportError as e:
        log.warning("RAGAS not available (%s) — skipping RAGAS eval", e)
        return {"error": f"missing dependency: {e}"}

    from generator import get_generator
    from reranker import get_reranker

    retriever = get_retriever()
    reranker  = get_reranker()
    generator = get_generator()

    # Check Ollama
    if not await generator.check_ollama():
        return {"error": "Ollama not running"}

    samples = dataset[:n_samples]
    print(f"\n{'='*65}")
    print(f"  RAGAS EVALUATION  ({len(samples)} samples)")
    print(f"{'='*65}")

    questions   = []
    answers     = []
    contexts    = []
    ground_truths = []

    for i, item in enumerate(samples, 1):
        query = item["query"]
        print(f"  [{i}/{len(samples)}] {query[:60]}")

        try:
            candidates = retriever.retrieve_sync(query, top_k=cfg.retrieval.top_k)
            ranked     = reranker.rerank_sync(query, candidates)
            response   = await generator.generate(query, ranked)

            # Context = top-3 text chunks (not graph)
            ctx_chunks = [
                r["text"] for r in ranked
                if r.get("source") != "graph"
            ][:3]

            questions.append(query)
            answers.append(response)
            contexts.append(ctx_chunks)
            ground_truths.append(item.get("expected_answer", ""))
        except Exception as e:
            log.warning("RAGAS sample failed for %r: %s", query, e)
            continue

    if not questions:
        return {"error": "no samples completed"}

    # Build RAGAS dataset
    ragas_data = Dataset.from_dict({
        "question":    questions,
        "answer":      answers,
        "contexts":    contexts,
        "ground_truth": ground_truths,
    })

    try:
        # Uses OPENAI_API_KEY from .env
        try:
            from ragas.embeddings import OpenAIEmbeddings as RagasOpenAIEmbeddings  # type: ignore
            embeddings = RagasOpenAIEmbeddings(model="text-embedding-3-small")
            score = ragas_evaluate(
                ragas_data,
                metrics=[faithfulness, answer_relevancy, context_recall],
                embeddings=embeddings,
            )
        except Exception:
            # fallback without explicit embeddings
            score = ragas_evaluate(
                ragas_data,
                metrics=[faithfulness, answer_relevancy, context_recall],
            )
        result = {
            "faithfulness":     round(float(score["faithfulness"]), 4),
            "answer_relevancy": round(float(score["answer_relevancy"]), 4),
            "context_recall":   round(float(score["context_recall"]), 4),
            "n_samples":        len(questions),
        }
        print(f"\n  RAGAS Results:")
        print(f"    Faithfulness:     {result['faithfulness']:.4f}")
        print(f"    Answer Relevancy: {result['answer_relevancy']:.4f}")
        print(f"    Context Recall:   = {r.get('context_recall', 0):.4f}")
        print(f"    Context Recall:   {result['context_recall']:.4f}")
        return result
    except Exception as e:
        log.error("RAGAS evaluate failed: %s", e)
        return {"error": str(e)}


# ── Latency benchmarking ──────────────────────────────────────────────────────

async def run_latency_eval(
    dataset: list[dict],
    n_samples: int = 10,
    verbose: bool = True,
) -> dict:
    """
    Benchmark end-to-end latency including TTFT.
    Target: TTFT < 2000ms.
    """
    from generator import get_generator
    from reranker import get_reranker

    retriever = get_retriever()
    reranker  = get_reranker()
    generator = get_generator()

    if not await generator.check_ollama():
        return {"error": "Ollama not running"}

    samples = dataset[:n_samples]
    print(f"\n{'='*65}")
    print(f"  LATENCY BENCHMARKING  ({len(samples)} queries)")
    print(f"  Target: TTFT < 2000ms")
    print(f"{'='*65}")

    ttft_times   = []
    total_times  = []
    retrieve_times = []

    for i, item in enumerate(samples, 1):
        query = item["query"]
        try:
            # Retrieval latency
            t0         = time.perf_counter()
            candidates = retriever.retrieve_sync(query)
            ranked     = reranker.rerank_sync(query, candidates)
            t_retrieve = (time.perf_counter() - t0) * 1000
            retrieve_times.append(t_retrieve)

            # TTFT — time until first token from generator
            t_gen  = time.perf_counter()
            ttft   = None
            total  = 0
            async for token in generator.stream(query, ranked):
                if ttft is None:
                    ttft = (time.perf_counter() - t_gen) * 1000
                total += len(token)
            total_time = (time.perf_counter() - t_gen) * 1000

            ttft_times.append(ttft or 0)
            total_times.append(total_time)

            status = "✅" if (ttft or 0) < 2000 else "❌"
            if verbose:
                print(f"  {i:2}. {status} TTFT={ttft:.0f}ms  "
                      f"total={total_time:.0f}ms  "
                      f"retrieve={t_retrieve:.0f}ms  "
                      f"{query[:45]}")
        except Exception as e:
            log.warning("Latency eval failed for %r: %s", query, e)

    if not ttft_times:
        return {"error": "no samples completed"}

    def pct(arr, p):
        return round(sorted(arr)[int(len(arr)*p/100)], 1)

    result = {
        "ttft": {
            "p50":  pct(ttft_times, 50),
            "p90":  pct(ttft_times, 90),
            "p95":  pct(ttft_times, 95),
            "mean": round(sum(ttft_times)/len(ttft_times), 1),
            "pass_rate": round(sum(1 for t in ttft_times if t < 2000)/len(ttft_times), 4),
        },
        "total_latency": {
            "p50":  pct(total_times, 50),
            "p90":  pct(total_times, 90),
            "mean": round(sum(total_times)/len(total_times), 1),
        },
        "retrieval_latency": {
            "mean": round(sum(retrieve_times)/len(retrieve_times), 1),
        },
        "n_samples": len(ttft_times),
    }

    print(f"\n  TTFT Results:")
    print(f"    p50={result['ttft']['p50']}ms  "
          f"p90={result['ttft']['p90']}ms  "
          f"p95={result['ttft']['p95']}ms")
    print(f"    Pass rate (< 2000ms): {result['ttft']['pass_rate']*100:.1f}%")
    return result


# ── Guardrail evaluation ──────────────────────────────────────────────────────

def run_guardrail_eval(verbose: bool = True) -> dict:
    """Run the built-in guardrail test suite from guardrails.py."""
    print(f"\n{'='*65}")
    print(f"  GUARDRAIL ROBUSTNESS TESTING")
    print(f"{'='*65}")
    result = run_test_suite(verbose=verbose)
    return {
        "pass_rate":  round(result["passed"] / result["total"], 4),
        "passed":     result["passed"],
        "failed":     result["failed"],
        "total":      result["total"],
    }


# ── Full evaluation runner ────────────────────────────────────────────────────

async def run_full_eval(
    retrieval_only: bool = False,
    latency_only:   bool = False,
    guardrails_only: bool = False,
    output_path:    str | None = None,
    n_ragas:        int = 10,
    n_latency:      int = 10,
    verbose:        bool = True,
) -> dict:
    results = {
        "timestamp":  datetime.utcnow().isoformat(),
        "config": {
            "embed_model":   cfg.embedding.model,
            "reranker":      cfg.reranker.model,
            "llm":           cfg.llm.model,
            "top_k":         cfg.retrieval.top_k,
            "rerank_top_k":  cfg.reranker.top_k,
        },
    }

    if guardrails_only:
        results["guardrails"] = run_guardrail_eval(verbose=verbose)
        _print_summary(results)
        _save_results(results, output_path)
        return results

    if latency_only:
        results["latency"] = await run_latency_eval(EVAL_DATASET, n_latency, verbose)
        _print_summary(results)
        _save_results(results, output_path)
        return results

    # Retrieval eval (always runs)
    results["retrieval"] = await run_retrieval_eval(EVAL_DATASET, verbose=verbose)

    if retrieval_only:
        results["guardrails"] = run_guardrail_eval(verbose=verbose)
        _print_summary(results)
        _save_results(results, output_path)
        return results

    # Full eval: also runs RAGAS + latency
    results["ragas"]      = await run_ragas_eval(EVAL_DATASET, n_ragas, verbose)
    results["latency"]    = await run_latency_eval(EVAL_DATASET, n_latency, verbose)
    results["guardrails"] = run_guardrail_eval(verbose=verbose)

    _print_summary(results)
    _save_results(results, output_path)
    return results


# ── Summary printer ───────────────────────────────────────────────────────────

def _print_summary(results: dict) -> None:
    print(f"\n{'='*65}")
    print(f"  EVALUATION SUMMARY")
    print(f"{'='*65}")

    if "retrieval" in results:
        r = results["retrieval"]
        print(f"  Retrieval:")
        for k, v in r.get("recall_at_k", {}).items():
            print(f"    Recall@{k:<3} = {v:.4f}")
        print(f"    MRR        = {r.get('mrr', 0):.4f}")
        print(f"    nDCG@5     = {r.get('ndcg_at_5', 0):.4f}")

    if "ragas" in results and "error" not in results["ragas"]:
        r = results["ragas"]
        print(f"  RAGAS:")
        print(f"    Faithfulness     = {r.get('faithfulness', 0):.4f}")
        print(f"    Answer Relevancy = {r.get('answer_relevancy', 0):.4f}")

    if "latency" in results and "error" not in results["latency"]:
        r = results["latency"]
        ttft = r.get("ttft", {})
        print(f"  Latency (TTFT):")
        print(f"    p50={ttft.get('p50',0)}ms  p90={ttft.get('p90',0)}ms  "
              f"pass={ttft.get('pass_rate',0)*100:.1f}%")

    if "guardrails" in results:
        r = results["guardrails"]
        print(f"  Guardrails:")
        print(f"    Pass rate = {r.get('pass_rate',0)*100:.1f}%  "
              f"({r.get('passed',0)}/{r.get('total',0)})")

    print(f"{'='*65}\n")


def _save_results(results: dict, output_path: str | None) -> None:
    out = Path(output_path or cfg.eval.output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_json(results, out)

    # Also save a clean summary
    summary_path = out.parent / "eval_summary.json"
    summary = {
        "timestamp": results.get("timestamp"),
        "recall_at_1":        results.get("retrieval", {}).get("recall_at_k", {}).get(1, 0),
        "recall_at_5":        results.get("retrieval", {}).get("recall_at_k", {}).get(5, 0),
        "mrr":                results.get("retrieval", {}).get("mrr", 0),
        "ndcg_at_5":          results.get("retrieval", {}).get("ndcg_at_5", 0),
        "faithfulness":       results.get("ragas", {}).get("faithfulness", None),
        "context_recall":     results.get("ragas", {}).get("context_recall", None),
        "answer_relevancy":   results.get("ragas", {}).get("answer_relevancy", None),
        "ttft_p50":           results.get("latency", {}).get("ttft", {}).get("p50", None),
        "ttft_pass_rate":     results.get("latency", {}).get("ttft", {}).get("pass_rate", None),
        "guardrail_pass_rate": results.get("guardrails", {}).get("pass_rate", None),
    }
    save_json(summary, summary_path)
    print(f"  Results saved → {out}")
    print(f"  Summary saved → {summary_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluation suite for UB CSE Chatbot")
    parser.add_argument("--retrieval-only",  action="store_true",
                        help="Only run retrieval metrics (no Ollama needed)")
    parser.add_argument("--latency-only",    action="store_true",
                        help="Only run latency benchmarks")
    parser.add_argument("--guardrails-only", action="store_true",
                        help="Only run guardrail tests (no indexes needed)")
    parser.add_argument("--output",          type=str, default=None,
                        help="Output path for results JSON")
    parser.add_argument("--n-ragas",         type=int, default=len(EVAL_DATASET),
                        help="Number of RAGAS samples (default 10)")
    parser.add_argument("--n-latency",       type=int, default=10,
                        help="Number of latency samples (default 10)")
    parser.add_argument("--quiet",           action="store_true",
                        help="Suppress per-query output")
    args = parser.parse_args()

    asyncio.run(run_full_eval(
        retrieval_only=args.retrieval_only,
        latency_only=args.latency_only,
        guardrails_only=args.guardrails_only,
        output_path=args.output,
        n_ragas=args.n_ragas,
        n_latency=args.n_latency,
        verbose=not args.quiet,
    ))


if __name__ == "__main__":
    main()