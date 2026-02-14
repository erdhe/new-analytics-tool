import io
import re
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="Sheet Copilot (MVP)", layout="wide")

def read_uploaded_file(uploaded_file):
    name = uploaded_file.name.lower()
    raw = uploaded_file.read()
    bio = io.BytesIO(raw)

    if name.endswith(".csv"):
        df = pd.read_csv(bio)
        return {"Sheet1": df}
    if name.endswith(".xlsx") or name.endswith(".xls"):
        xls = pd.ExcelFile(bio)
        return {s: pd.read_excel(xls, sheet_name=s) for s in xls.sheet_names}

    raise ValueError("Unsupported file type. Upload .csv or .xlsx")

def detect_date_cols(df):
    date_cols = []
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            date_cols.append(c)
            continue
        if df[c].dtype == "object":
            sample = df[c].dropna().astype(str).head(30)
            if sample.empty:
                continue
            parsed = pd.to_datetime(sample, errors="coerce", utc=False, infer_datetime_format=True)
            if parsed.notna().mean() >= 0.6:
                date_cols.append(c)
    return date_cols

def coerce_dates(df, cols):
    for c in cols:
        df[c] = pd.to_datetime(df[c], errors="coerce", infer_datetime_format=True)
    return df

def profile_df(df: pd.DataFrame):
    out = {}
    out["rows"] = int(df.shape[0])
    out["cols"] = int(df.shape[1])
    out["missing_cells"] = int(df.isna().sum().sum())
    out["duplicate_rows"] = int(df.duplicated().sum())
    out["dtypes"] = df.dtypes.astype(str)

    missing_by_col = (df.isna().mean() * 100).round(2).sort_values(ascending=False)
    out["missing_by_col"] = missing_by_col

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    out["numeric_cols"] = numeric_cols

    cat_cols = [c for c in df.columns if c not in numeric_cols]
    out["cat_cols"] = cat_cols

    if numeric_cols:
        desc = df[numeric_cols].describe().T
        desc["missing_%"] = (df[numeric_cols].isna().mean() * 100).round(2)
        out["numeric_summary"] = desc
    else:
        out["numeric_summary"] = pd.DataFrame()

    if cat_cols:
        card = pd.Series({c: df[c].nunique(dropna=True) for c in cat_cols}).sort_values(ascending=False)
        out["cardinality"] = card
    else:
        out["cardinality"] = pd.Series(dtype="int")

    return out

def top_outliers(df, num_col, n=10):
    s = df[num_col]
    if s.dropna().empty:
        return pd.DataFrame()
    z = (s - s.mean()) / (s.std(ddof=0) if s.std(ddof=0) != 0 else 1)
    out = df.loc[z.abs().sort_values(ascending=False).head(n).index].copy()
    out["_zscore_abs"] = z.abs().loc[out.index]
    return out.sort_values("_zscore_abs", ascending=False)

def correlations(df, numeric_cols):
    if len(numeric_cols) < 2:
        return pd.DataFrame()
    corr = df[numeric_cols].corr(numeric_only=True)
    pairs = []
    cols = corr.columns
    for i in range(len(cols)):
        for j in range(i+1, len(cols)):
            pairs.append((cols[i], cols[j], corr.iloc[i, j]))
    res = pd.DataFrame(pairs, columns=["col_a", "col_b", "corr"]).dropna()
    res["abs_corr"] = res["corr"].abs()
    return res.sort_values("abs_corr", ascending=False)

def time_trend(df, date_col, value_col, freq="M"):
    tmp = df[[date_col, value_col]].dropna().copy()
    if tmp.empty:
        return None
    tmp = tmp.sort_values(date_col)
    grp = tmp.set_index(date_col)[value_col].resample(freq).sum()
    out = grp.reset_index().rename(columns={value_col: "value"})
    return out

def similarity(a, b):
    a, b = a.strip(), b.strip()
    if a == b:
        return 1.0
    ta = set(re.findall(r"[a-z0-9]+", a.lower()))
    tb = set(re.findall(r"[a-z0-9]+", b.lower()))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)

def answer_question(df, q):
    ql = q.strip().lower()

    m = re.search(r"top\s+(\d+)\s+by\s+(.+)", ql)
    if m:
        n = int(m.group(1))
        metric = m.group(2).strip()
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not num_cols:
            return ("I couldn't find numeric columns to rank by.", None)
        best = max(num_cols, key=lambda c: similarity(c, metric))
        view_cols = [c for c in df.columns if c != best][:3] + [best]
        res = df[view_cols].dropna(subset=[best]).sort_values(best, ascending=False).head(n)
        return (f"Top {n} rows by **{best}** (matched from '{metric}').", res)

    if "missing" in ql:
        miss = (df.isna().mean() * 100).round(2).sort_values(ascending=False)
        res = miss.reset_index()
        res.columns = ["column", "missing_%"]
        return ("Missing values by column (%).", res)

    if "duplicate" in ql:
        dups = df[df.duplicated(keep=False)]
        return (f"Found **{int(df.duplicated().sum())}** duplicate rows (showing all duplicate groups).", dups.head(200))

    if "correlation" in ql or "correlate" in ql:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        corr = correlations(df, num_cols).head(20)
        return ("Top correlations (absolute) among numeric columns.", corr)

    return ("MVP Q&A supports: top-N rankings, missing values, duplicates, correlations. "
            "Next step: add LLM-based question parsing + citations.", None)

st.title("📊 Sheet Copilot (Browser MVP)")
st.caption("Upload a spreadsheet, get profiling + insights, and ask basic questions.")

uploaded = st.file_uploader("Upload a spreadsheet (.xlsx / .csv)", type=["xlsx", "xls", "csv"])
if not uploaded:
    st.info("Upload a file to start.")
    st.stop()

try:
    sheets = read_uploaded_file(uploaded)
except Exception as e:
    st.error(str(e))
    st.stop()

sheet_names = list(sheets.keys())
sheet = st.selectbox("Select sheet", sheet_names)
df = sheets[sheet].copy()
# Auto-clean numeric-looking columns
for col in df.columns:
    if df[col].dtype == "object":
        # Remove currency symbols, commas, spaces
        cleaned = (
            df[col]
            .astype(str)
            .str.replace(r"[^\d\.\-]", "", regex=True)
        )
        # Try converting to numeric
        converted = pd.to_numeric(cleaned, errors="coerce")
        # If at least 60% can be converted → treat as numeric
        if converted.notna().mean() > 0.6:
            df[col] = converted

date_cols = detect_date_cols(df)
df = coerce_dates(df, date_cols)

tabs = st.tabs(["Data", "Profile", "Insights", "Ask"])

with tabs[0]:
    st.subheader("Preview")
    st.dataframe(df.head(200), use_container_width=True)
    st.caption(f"Rows: {df.shape[0]:,} | Columns: {df.shape[1]:,}")

with tabs[1]:
    st.subheader("Profile")
    prof = profile_df(df)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{prof['rows']:,}")
    c2.metric("Columns", f"{prof['cols']:,}")
    c3.metric("Missing cells", f"{prof['missing_cells']:,}")
    c4.metric("Duplicate rows", f"{prof['duplicate_rows']:,}")

    st.markdown("#### Missing values by column (%)")
    st.dataframe(
        prof["missing_by_col"].reset_index().rename(columns={"index": "column", 0: "missing_%"}),
        use_container_width=True
    )

    st.markdown("#### Numeric summary")
    if not prof["numeric_summary"].empty:
        st.dataframe(prof["numeric_summary"], use_container_width=True)
    else:
        st.info("No numeric columns detected.")

    st.markdown("#### Categorical column cardinality (unique values)")
    if len(prof["cardinality"]) > 0:
        st.dataframe(
            prof["cardinality"].reset_index().rename(columns={"index": "column", 0: "unique_values"}),
            use_container_width=True
        )
    else:
        st.info("No categorical columns detected.")

with tabs[2]:
    st.subheader("Auto insights")

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not num_cols:
        st.warning("No numeric columns found — insights are limited.")
    else:
        st.markdown("#### Correlations (top 15)")
        corr = correlations(df, num_cols).head(15)
        if corr.empty:
            st.info("Not enough numeric columns for correlation.")
        else:
            st.dataframe(corr[["col_a", "col_b", "corr"]], use_container_width=True)

        st.markdown("#### Outliers")
        col = st.selectbox("Pick a metric to find outliers", num_cols)
        out = top_outliers(df, col, n=10)
        if out.empty:
            st.info("No outliers found (or no data).")
        else:
            st.dataframe(out.head(50), use_container_width=True)

        if date_cols:
            st.markdown("#### Time trend")
            dcol = st.selectbox("Date column", date_cols)
            vcol = st.selectbox("Value column", num_cols, index=0)
            freq = st.selectbox("Granularity", ["D", "W", "M"], index=2)
            trend = time_trend(df, dcol, vcol, freq=freq)
            if trend is None or trend.empty:
                st.info("No trend available (missing date/value data).")
            else:
                fig = px.line(trend, x=dcol, y="value", title=f"{vcol} over time ({freq})")
                st.plotly_chart(fig, use_container_width=True)

with tabs[3]:
    st.subheader("Ask a question")
    st.caption("Try: 'top 10 by revenue' / 'missing values' / 'duplicates' / 'correlations'")
    q = st.text_input("Your question")
    if q:
        text, table = answer_question(df, q)
        st.markdown(text)
        if table is not None and isinstance(table, pd.DataFrame):
            st.dataframe(table, use_container_width=True)
