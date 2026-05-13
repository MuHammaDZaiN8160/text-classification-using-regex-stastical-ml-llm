import streamlit as st
import httpx
import pandas as pd

API_URL = "https://text-classification-using-regex-stastical-ml-llm-production.up.railway.app"

st.set_page_config(page_title="Log Classifier", page_icon="🔍", layout="wide")
st.title("Log Classification System")
st.caption("Hybrid approach: Regex → BERT + Logistic Regression → LLM")

METHOD_COLORS = {
    "regex": "🟢",
    "bert": "🔵",
    "llm": "🟣",
}

LABEL_COLORS = {
    "Security Alert": "🔴",
    "Resource Usage": "🟡",
    "Workflow Error": "🟠",
}

tab1, tab2, tab3 = st.tabs(["Classify", "Batch Classify", "Train Model"])

# ── Tab 1: Single classify ─────────────────────────────────────────────────
with tab1:
    st.subheader("Single Log Classification")
    log_text = st.text_area(
        "Enter a log message:",
        height=100,
        placeholder="e.g. Multiple login failures occurred on user 9052 account",
    )

    if st.button("Classify", type="primary"):
        if not log_text.strip():
            st.warning("Please enter a log message.")
        else:
            with st.spinner("Classifying..."):
                try:
                    resp = httpx.post(f"{API_URL}/classify", json={"text": log_text}, timeout=30)
                    if resp.status_code == 200:
                        data = resp.json()
                        label = data["label"]
                        method = data["method"]
                        confidence = data.get("confidence")

                        st.success("Classification complete")
                        col_a, col_b, col_c = st.columns(3)
                        col_a.metric("Label", f"{LABEL_COLORS.get(label, '⚪')} {label}")
                        col_b.metric("Method Used", f"{METHOD_COLORS.get(method, '⚪')} {method.upper()}")
                        if confidence is not None:
                            col_c.metric("Confidence", f"{confidence * 100:.1f}%")
                        else:
                            col_c.metric("Confidence", "N/A (LLM)")
                    else:
                        st.error(f"API error: {resp.text}")
                except httpx.ConnectError:
                    st.error("Cannot connect to the API. Make sure the backend is running on port 8000.")

# ── Tab 2: Batch classify ──────────────────────────────────────────────────
with tab2:
    st.subheader("Batch Classification")
    st.info("Enter one log message per line.")
    batch_text = st.text_area("Log messages (one per line):", height=200)

    if st.button("Classify All", type="primary"):
        lines = [l.strip() for l in batch_text.strip().splitlines() if l.strip()]
        if not lines:
            st.warning("Please enter at least one log message.")
        else:
            with st.spinner(f"Classifying {len(lines)} messages..."):
                try:
                    resp = httpx.post(f"{API_URL}/classify/batch", json=lines, timeout=60)
                    if resp.status_code == 200:
                        results = resp.json()
                        rows = [
                            {
                                "Log Message": lines[i],
                                "Label": r["label"],
                                "Method": r["method"],
                                "Confidence": f"{r['confidence']*100:.1f}%" if r.get("confidence") is not None else "N/A",
                            }
                            for i, r in enumerate(results)
                        ]
                        df = pd.DataFrame(rows)
                        st.dataframe(df, use_container_width=True)

                        st.markdown("**Label Distribution**")
                        counts = df["Label"].value_counts().reset_index()
                        counts.columns = ["Label", "Count"]
                        st.bar_chart(counts.set_index("Label"))

                        st.markdown("**Method Distribution**")
                        method_counts = df["Method"].value_counts().reset_index()
                        method_counts.columns = ["Method", "Count"]
                        st.bar_chart(method_counts.set_index("Method"))
                    else:
                        st.error(f"API error: {resp.text}")
                except httpx.ConnectError:
                    st.error("Cannot connect to the API. Make sure the backend is running on port 8000.")

# ── Tab 3: Train model ─────────────────────────────────────────────────────
with tab3:
    st.subheader("Train BERT + Logistic Regression Model")

    col_stats, col_train = st.columns(2)

    with col_stats:
        st.markdown("**Current Model Status**")
        if st.button("Refresh Stats"):
            try:
                resp = httpx.get(f"{API_URL}/stats", timeout=10)
                if resp.status_code == 200:
                    stats = resp.json()
                    trained = stats.get("bert_trained", False)
                    st.markdown(f"**BERT Trained:** {'Yes' if trained else 'No'}")
                    st.markdown(f"**Min Samples Required:** {stats.get('min_samples_required', 10)}")
                    counts = stats.get("class_counts", {})
                    if counts:
                        st.markdown("**Samples per Class:**")
                        for cls, cnt in counts.items():
                            st.markdown(f"- {LABEL_COLORS.get(cls,'⚪')} {cls}: **{cnt}**")
                else:
                    st.error(resp.text)
            except httpx.ConnectError:
                st.error("Cannot connect to the API.")

    with col_train:
        st.markdown("**Option 1 — Train on built-in sample data**")
        st.caption("Uses `data/sample_logs.csv` (33 labeled rows).")
        if st.button("Train on Sample Data", type="primary"):
            with st.spinner("Training... this may take a minute."):
                try:
                    resp = httpx.post(f"{API_URL}/train", json={}, timeout=300)
                    if resp.status_code == 200:
                        st.success(resp.json()["message"])
                    else:
                        st.error(resp.text)
                except httpx.ConnectError:
                    st.error("Cannot connect to the API.")

        st.markdown("---")
        st.markdown("**Option 2 — Upload your own CSV**")
        st.caption("CSV must have columns: `log_message`, `label`")

        uploaded = st.file_uploader("Choose a CSV file", type=["csv"])
        if uploaded is not None:
            preview_df = pd.read_csv(uploaded)
            st.markdown(f"Preview — **{len(preview_df)} rows**")
            st.dataframe(preview_df.head(5), use_container_width=True)

            if "log_message" not in preview_df.columns or "label" not in preview_df.columns:
                st.error("CSV must have `log_message` and `label` columns.")
            else:
                if st.button("Train on Uploaded CSV", type="primary"):
                    with st.spinner("Training..."):
                        try:
                            uploaded.seek(0)
                            resp = httpx.post(
                                f"{API_URL}/train/upload",
                                files={"file": (uploaded.name, uploaded, "text/csv")},
                                timeout=300,
                            )
                            if resp.status_code == 200:
                                result = resp.json()
                                st.success(result["message"])
                                st.json(result.get("class_counts", {}))
                            else:
                                st.error(resp.text)
                        except httpx.ConnectError:
                            st.error("Cannot connect to the API.")

# ── Sidebar ────────────────────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.markdown("**Classification Pipeline**")
st.sidebar.markdown("🟢 **Regex** — fixed patterns · confidence 100%")
st.sidebar.markdown("🔵 **BERT** — variable patterns · shows confidence %")
st.sidebar.markdown("🟣 **LLM** — complex/rare patterns · via Groq")
st.sidebar.markdown("---")
st.sidebar.markdown("**Labels**")
st.sidebar.markdown("🔴 Security Alert")
st.sidebar.markdown("🟡 Resource Usage")
st.sidebar.markdown("🟠 Workflow Error")
st.sidebar.markdown("---")
st.sidebar.markdown("**CSV Format**")
st.sidebar.code("log_message,label\nLogin failed for user,Security Alert")
