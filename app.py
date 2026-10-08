import streamlit as st

from typing import TypedDict, List, Optional

from langchain_core.messages import BaseMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Interview Preparation Agent",
    page_icon="🎯",
    layout="centered"
)


# ============================================================
# 2. API KEY
# ============================================================

api_key = st.secrets.get("GEMINI_API_KEY")

if not api_key:
    st.error("GEMINI_API_KEY is not configured.")
    st.stop()


# ============================================================
# 3. LLM
# ============================================================

llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite-preview",
    google_api_key=api_key,
    temperature=0.7
)


# ============================================================
# 4. STATE
# ============================================================

class InterviewState(TypedDict):
    messages: List[BaseMessage]

    candidate_name: Optional[str]
    target_role: Optional[str]
    experience_level: Optional[str]

    current_question: Optional[str]
    current_answer: Optional[str]

    question_number: int
    total_questions: int

    evaluation: Optional[str]
    final_report: Optional[str]


# ============================================================
# 5. QUESTION GENERATOR
# ============================================================

@tool
def generate_interview_question(
    role: str,
    experience_level: str,
    question_number: int
) -> str:

    prompt = f"""
You are a professional technical interviewer.

Candidate Role:
{role}

Experience Level:
{experience_level}

Question Number:
{question_number}

Generate ONE realistic interview question.

Requirements:
- Match the candidate's experience level.
- Make it relevant to the target role.
- Mix technical, practical, scenario-based and behavioral
  questions across the interview.
- Do not provide the answer.
- Do not provide explanation.
- Return ONLY the interview question.
"""

    response = llm.invoke(prompt)

    content = response.content

    if isinstance(content, list):
        parts = []

        for item in content:
            if isinstance(item, dict):
                parts.append(item.get("text", ""))
            else:
                parts.append(str(item))

        return " ".join(parts).strip()

    return str(content).strip()


# ============================================================
# 6. ANSWER EVALUATOR
# ============================================================

@tool
def evaluate_interview_answer(
    role: str,
    question: str,
    answer: str
) -> str:

    prompt = f"""
You are a senior professional interviewer.

Target Role:
{role}

Interview Question:
{question}

Candidate Answer:
{answer}

Evaluate the candidate's answer.

Return exactly this format:

SCORE: X/10

STRENGTHS:
- Point 1
- Point 2

WEAKNESSES:
- Point 1
- Point 2

IMPROVEMENTS:
- Point 1
- Point 2

IDEAL ANSWER POINTS:
- Point 1
- Point 2
- Point 3

INTERVIEWER FEEDBACK:
Short practical feedback.

Focus on:
- Correctness
- Technical knowledge
- Relevance
- Problem solving
- Communication
- Practical understanding

Be constructive and professional.
"""

    response = llm.invoke(prompt)

    content = response.content

    if isinstance(content, list):
        parts = []

        for item in content:
            if isinstance(item, dict):
                parts.append(item.get("text", ""))
            else:
                parts.append(str(item))

        return "\n".join(parts).strip()

    return str(content).strip()


# ============================================================
# 7. FINAL REPORT
# ============================================================

def generate_final_report(
    candidate_name,
    role,
    experience,
    question_count,
    evaluations
):

    evaluation_text = "\n\n".join(evaluations)

    prompt = f"""
You are an expert interview coach.

Candidate:
{candidate_name}

Target Role:
{role}

Experience Level:
{experience}

Questions Completed:
{question_count}

Interview Evaluations:
{evaluation_text}

Create a final interview preparation report.

Include:

1. Overall Assessment
2. Technical Performance
3. Communication Performance
4. Problem Solving Performance
5. Strong Areas
6. Weak Areas
7. Topics To Revise
8. Interview Tips
9. Recommended Preparation Plan
10. Final Readiness Rating out of 10

Keep it practical and constructive.
"""

    response = llm.invoke(prompt)

    content = response.content

    if isinstance(content, list):
        parts = []

        for item in content:
            if isinstance(item, dict):
                parts.append(item.get("text", ""))
            else:
                parts.append(str(item))

        return "\n".join(parts).strip()

    return str(content).strip()


# ============================================================
# 8. STREAMLIT UI
# ============================================================

st.title("🎯 AI Interview Preparation Agent")

st.write(
    "Practice realistic interview questions, "
    "receive AI feedback, and improve your interview performance."
)


# ------------------------------------------------------------
# Candidate Information
# ------------------------------------------------------------

with st.sidebar:

    st.header("Interview Setup")

    candidate_name = st.text_input(
        "Candidate Name"
    )

    target_role = st.text_input(
        "Job Role",
        value="Embedded Systems Engineer"
    )

    experience_level = st.selectbox(
        "Experience Level",
        [
            "Fresher",
            "Junior",
            "Mid-Level",
            "Senior"
        ]
    )

    total_questions = st.slider(
        "Number of Questions",
        min_value=1,
        max_value=10,
        value=5
    )


# ============================================================
# SESSION STATE
# ============================================================

if "started" not in st.session_state:
    st.session_state.started = False

if "question_number" not in st.session_state:
    st.session_state.question_number = 0

if "current_question" not in st.session_state:
    st.session_state.current_question = None

if "evaluations" not in st.session_state:
    st.session_state.evaluations = []

if "finished" not in st.session_state:
    st.session_state.finished = False


# ============================================================
# START INTERVIEW
# ============================================================

if not st.session_state.started:

    if st.button(
        "🚀 Start Interview",
        use_container_width=True
    ):

        if not candidate_name.strip():
            st.warning("Please enter your name.")

        elif not target_role.strip():
            st.warning("Please enter a job role.")

        else:

            st.session_state.started = True
            st.session_state.question_number = 1

            question = generate_interview_question.invoke({
                "role": target_role,
                "experience_level": experience_level,
                "question_number": 1
            })

            st.session_state.current_question = question

            st.rerun()


# ============================================================
# INTERVIEW
# ============================================================

if (
    st.session_state.started
    and not st.session_state.finished
):

    st.progress(
        st.session_state.question_number / total_questions
    )

    st.subheader(
        f"Question "
        f"{st.session_state.question_number}/"
        f"{total_questions}"
    )

    st.info(
        st.session_state.current_question
    )

    answer = st.text_area(
        "Your Answer",
        height=180,
        key=f"answer_{st.session_state.question_number}"
    )

    if st.button(
        "Submit Answer",
        use_container_width=True
    ):

        if not answer.strip():

            st.warning(
                "Please enter your answer."
            )

        else:

            evaluation = evaluate_interview_answer.invoke({
                "role": target_role,
                "question": st.session_state.current_question,
                "answer": answer
            })

            st.session_state.evaluations.append(
                evaluation
            )

            st.subheader(
                "📊 AI Evaluation"
            )

            st.markdown(evaluation)

            if (
                st.session_state.question_number
                < total_questions
            ):

                st.session_state.question_number += 1

                next_question = (
                    generate_interview_question.invoke({
                        "role": target_role,
                        "experience_level": experience_level,
                        "question_number":
                            st.session_state.question_number
                    })
                )

                st.session_state.current_question = (
                    next_question
                )

                st.rerun()

            else:

                st.session_state.finished = True
                st.rerun()


# ============================================================
# FINAL REPORT
# ============================================================

if st.session_state.finished:

    st.success(
        "🎉 Interview completed successfully!"
    )

    if st.button(
        "Generate Final Report",
        use_container_width=True
    ):

        final_report = generate_final_report(
            candidate_name,
            target_role,
            experience_level,
            total_questions,
            st.session_state.evaluations
        )

        st.subheader(
            "🏆 Final Interview Report"
        )

        st.markdown(final_report)

    if st.button(
        "🔄 Start New Interview",
        use_container_width=True
    ):

        for key in [
            "started",
            "question_number",
            "current_question",
            "evaluations",
            "finished"
        ]:
            if key in st.session_state:
                del st.session_state[key]

        st.rerun()