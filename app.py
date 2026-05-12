import time
import pandas as pd
import streamlit as st

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


st.set_page_config(
    page_title="GAP Early Payoff AI Assistant",
    page_icon="🚗",
    layout="wide"
)


@st.cache_data
def load_data():
    df = pd.read_csv("gold_gap_ai_context.csv")

    df["GAP_AMOUNT_EUR"] = pd.to_numeric(df["GAP_AMOUNT_EUR"], errors="coerce").fillna(0)
    df["DAYS_TO_PAYOFF"] = pd.to_numeric(df["DAYS_TO_PAYOFF"], errors="coerce")

    return df


gold_df = load_data()


business_rules = """
A GAP early payoff case occurs when a vehicle loan with GAP protection is paid off before its expected maturity date.

A high-risk GAP early payoff case is a loan paid off within 90 days of the original loan date.

A medium-risk GAP early payoff case is a loan paid off between 91 and 180 days of the original loan date.

A low-risk GAP early payoff case is a loan paid off after 180 days.

A case requires manual review when the GAP amount is missing, the dealer GAP company is missing, the dealership name is missing, or the loan is paid off within 90 days.

Dealer risk should be reviewed when a dealership has repeated high-risk GAP early payoff cases.

Total GAP Amount means the sum of GAP_AMOUNT_EUR.

Total Accounts means the distinct count of ACCOUNT_NUMBER.
"""

data_dictionary = """
ACCOUNT_NUMBER is the synthetic account reference.
LOAN_ID is the loan identifier.
PAYOFF_DATE is the date the loan was paid off.
ORIGINAL_LOAN_DATE is the original loan date.
DAYS_TO_PAYOFF is the number of days between original loan date and payoff date.
GAP_TYPE identifies the GAP product category.
GAP_AMOUNT_EUR is the GAP amount in euros.
DEALERSHIP identifies the dealer linked to the loan.
DEALER_GAP_COMPANY identifies the GAP provider.
RISK_CATEGORY classifies the case as HIGH RISK, MEDIUM RISK, or LOW RISK.
RISK_REASON explains why the case was categorized.
AI_CONTEXT_TEXT is the plain-English row-level explanation used by the RAG assistant.
"""

dashboard_metrics = """
The GAP Early Payoff dashboard includes slicers for GAP Type, Loan Type, Payoff Date, Dealership, and Dealership Region.

The KPI cards show Total Accounts, Total GAP Amount, High Risk Count, and Review Required Count.

The line chart shows Total Accounts by Payoff Date.

The detail table shows Account Number, Loan ID, Original Loan Date, Payoff Date, Loan Type, Loan Type ID, GAP Type, Dealership, GAP Amount, Dealer GAP Company, Risk Category, and Risk Reason.

Dealer summary shows which dealerships have higher early payoff activity and higher risk counts.
"""


@st.cache_resource
def build_retriever(df):
    texts = []

    documents = {
        "gap_business_rules": business_rules,
        "gap_data_dictionary": data_dictionary,
        "gap_dashboard_metrics": dashboard_metrics,
    }

    for _, content in documents.items():
        chunks = [chunk.strip() for chunk in content.split("\n\n") if chunk.strip()]
        texts.extend(chunks)

    for _, row in df.iterrows():
        texts.append(str(row["AI_CONTEXT_TEXT"]))

    vectorizer = TfidfVectorizer(stop_words="english")
    text_vectors = vectorizer.fit_transform(texts)

    return texts, vectorizer, text_vectors


texts, vectorizer, text_vectors = build_retriever(gold_df)


def retrieve_context(question, top_k=5):
    question_vector = vectorizer.transform([question])
    similarity_scores = cosine_similarity(question_vector, text_vectors).flatten()
    top_indices = similarity_scores.argsort()[-top_k:][::-1]

    return [texts[idx] for idx in top_indices]


def query_data(question, df):
    question_lower = question.lower()

    if "top" in question_lower or "highest" in question_lower or "dealer" in question_lower or "dealership" in question_lower:
        dealer_summary = (
            df.groupby("DEALERSHIP")
            .agg(
                total_accounts=("ACCOUNT_NUMBER", "nunique"),
                total_gap_amount=("GAP_AMOUNT_EUR", "sum"),
                high_risk_count=("RISK_CATEGORY", lambda x: (x == "HIGH RISK").sum()),
            )
            .reset_index()
            .sort_values(["high_risk_count", "total_gap_amount"], ascending=False)
            .head(5)
        )

        return dealer_summary

    if "high risk" in question_lower:
        return df[df["RISK_CATEGORY"] == "HIGH RISK"][
            ["ACCOUNT_NUMBER", "LOAN_ID", "DEALERSHIP", "GAP_AMOUNT_EUR", "DAYS_TO_PAYOFF", "RISK_REASON"]
        ].head(10)

    if "total gap" in question_lower or "gap amount" in question_lower:
        total_gap = df["GAP_AMOUNT_EUR"].sum()
        return f"Total GAP Amount EUR: €{total_gap:,.2f}"

    if "total account" in question_lower or "how many account" in question_lower:
        total_accounts = df["ACCOUNT_NUMBER"].nunique()
        return f"Total Accounts: {total_accounts}"

    summary = {
        "total_accounts": int(df["ACCOUNT_NUMBER"].nunique()),
        "total_gap_amount": round(float(df["GAP_AMOUNT_EUR"].sum()), 2),
        "high_risk_count": int((df["RISK_CATEGORY"] == "HIGH RISK").sum()),
        "total_cases": int(len(df)),
    }

    return summary


def generate_answer(question, retrieved_context, data_result):
    context_text = "\n".join(retrieved_context[:3])

    if isinstance(data_result, pd.DataFrame):
        data_text = data_result.to_string(index=False)
    else:
        data_text = str(data_result)

    answer = f"""
**Question:** {question}

**Relevant business context:**  
{context_text}

**Data result:**  
{data_text}

**Business summary:**  
Based on the retrieved GAP early payoff rules and the Gold AI context data, this answer should be interpreted using the business rules above. High-risk cases are mainly driven by payoff within 90 days, missing GAP/company information, or repeated dealer-level early payoff activity.
"""

    return answer


st.title("GAP Early Payoff AI Assistant")
st.caption("Free RAG-style assistant using TF-IDF retrieval, cosine similarity, and Streamlit deployment.")

with st.sidebar:
    st.header("Filters")

    gap_type = st.selectbox(
        "GAP Type",
        ["All"] + sorted(gold_df["GAP_TYPE"].dropna().unique().tolist())
    )

    risk_category = st.selectbox(
        "Risk Category",
        ["All"] + sorted(gold_df["RISK_CATEGORY"].dropna().unique().tolist())
    )

    dealership = st.selectbox(
        "Dealership",
        ["All"] + sorted(gold_df["DEALERSHIP"].dropna().unique().tolist())
    )


filtered_df = gold_df.copy()

if gap_type != "All":
    filtered_df = filtered_df[filtered_df["GAP_TYPE"] == gap_type]

if risk_category != "All":
    filtered_df = filtered_df[filtered_df["RISK_CATEGORY"] == risk_category]

if dealership != "All":
    filtered_df = filtered_df[filtered_df["DEALERSHIP"] == dealership]


col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Accounts", filtered_df["ACCOUNT_NUMBER"].nunique())
col2.metric("Total GAP Amount", f"€{filtered_df['GAP_AMOUNT_EUR'].sum():,.2f}")
col3.metric("High Risk Cases", int((filtered_df["RISK_CATEGORY"] == "HIGH RISK").sum()))
col4.metric("Total Cases", len(filtered_df))


st.divider()

st.subheader("Ask the AI Assistant")

question = st.text_input(
    "Ask a question",
    placeholder="Example: Which dealerships have the highest early payoff risk?"
)

if st.button("Ask"):
    if question.strip():
        start = time.time()

        retrieved = retrieve_context(question, top_k=5)
        data_result = query_data(question, filtered_df)
        answer = generate_answer(question, retrieved, data_result)

        response_time = round(time.time() - start, 2)

        st.markdown(answer)
        st.caption(f"Response time: {response_time} seconds")
    else:
        st.warning("Please enter a question.")


st.divider()

st.subheader("GAP Early Payoff Records")

display_columns = [
    "ACCOUNT_NUMBER",
    "LOAN_ID",
    "DEALERSHIP",
    "GAP_TYPE",
    "GAP_AMOUNT_EUR",
    "DAYS_TO_PAYOFF",
    "RISK_CATEGORY",
    "RISK_REASON",
]

available_columns = [col for col in display_columns if col in filtered_df.columns]

st.dataframe(
    filtered_df[available_columns],
    use_container_width=True
)


st.divider()

st.subheader("Model Evaluation Results")

try:
    eval_df = pd.read_csv("model_evaluation_results.csv")
    st.dataframe(eval_df, use_container_width=True)

    if "score" in eval_df.columns:
        st.metric("Evaluation Accuracy", f"{eval_df['score'].mean() * 100:.1f}%")

except FileNotFoundError:
    st.info("model_evaluation_results.csv not found yet.")