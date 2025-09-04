# app.py
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
import plotly.express as px

# -----------------------
# SETTINGS
# -----------------------
st.set_page_config(
    page_title="NLP Model Metrics Dashboard",
    page_icon="📊",
    layout="wide"
)

# -----------------------
# SIDEBAR – INPUT
# -----------------------
st.sidebar.title("⚙️ Configurazione")
st.sidebar.markdown(
    "Carica un CSV con colonne: **model, accuracy, precision, recall, f1**.\n\n"
    "_Se non carichi nulla, uso il file di default._"
)

uploaded = st.sidebar.file_uploader("Carica CSV", type=["csv"], accept_multiple_files=False)

default_path = r"C:\Users\DAVIDEPEDRETTI\Desktop\NLP_news\NTBK_Transformers.ipynb"
if uploaded is not None:
    df = pd.read_csv(uploaded)
else:
    df = pd.read_csv(default_path)

# Validazione minima
required_cols = {"model", "accuracy", "precision", "recall", "f1"}
df.columns = [c.lower() for c in df.columns]
missing = required_cols - set(df.columns)
if missing:
    st.error(f"Il CSV deve contenere queste colonne: {sorted(required_cols)}.\nMancano: {sorted(missing)}")
    st.stop()

# Cast metriche a float (robusto)
for col in ["accuracy", "precision", "recall", "f1"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Rimuovi righe non valide
df = df.dropna(subset=["model", "accuracy", "precision", "recall", "f1"]).copy()

# -----------------------
# FILTRI / ORDINAMENTO
# -----------------------
st.sidebar.subheader("🔎 Filtri")
min_acc = st.sidebar.slider("Soglia minima accuracy", 0.0, 1.0, 0.0, 0.001)
min_f1  = st.sidebar.slider("Soglia minima F1", 0.0, 1.0, 0.0, 0.001)
df = df[(df["accuracy"] >= min_acc) & (df["f1"] >= min_f1)].reset_index(drop=True)

sort_by = st.sidebar.selectbox("Ordina per", ["f1", "accuracy", "precision", "recall"])
ascending = st.sidebar.toggle("Ordine crescente", value=False)

# -----------------------
# SCORE COMPOSITO
# -----------------------
st.sidebar.subheader("🧮 Pesi Score composito")
w_acc  = st.sidebar.slider("Peso Accuracy",  0.0, 1.0, 0.25, 0.05)
w_prec = st.sidebar.slider("Peso Precision", 0.0, 1.0, 0.25, 0.05)
w_rec  = st.sidebar.slider("Peso Recall",    0.0, 1.0, 0.25, 0.05)
w_f1   = st.sidebar.slider("Peso F1",        0.0, 1.0, 0.25, 0.05)
weights = np.array([w_acc, w_prec, w_rec, w_f1])
if weights.sum() == 0:
    weights = np.array([1, 1, 1, 1])
weights = weights / weights.sum()

metrics_cols = ["accuracy", "precision", "recall", "f1"]

# Normalizzazione 0-1 (min-max)
norm_df = df.copy()
for c in metrics_cols:
    cmin, cmax = norm_df[c].min(), norm_df[c].max()
    norm_df[c + "_norm"] = (norm_df[c] - cmin) / (cmax - cmin) if cmax > cmin else 1.0

norm_matrix = norm_df[[c + "_norm" for c in metrics_cols]].to_numpy()
df["score_composito"] = norm_matrix @ weights

# Ranking
df = df.sort_values(by=sort_by, ascending=ascending).reset_index(drop=True)
df["rank_"+sort_by] = np.arange(1, len(df) + 1)
df = df.sort_values(by="score_composito", ascending=False).reset_index(drop=True)
df["rank_score"] = np.arange(1, len(df) + 1)

# -----------------------
# HEADER
# -----------------------
st.title("📊 NLP Model Metrics Dashboard")
st.caption("Visualizzazione e confronto delle metriche di classificazione (accuracy, precision, recall, f1).")

if not df.empty:
    top_row = df.iloc[0]
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("🥇 Modello Top (Score)", str(top_row["model"]))
    c2.metric("Score composito", f"{top_row['score_composito']:.3f}")
    c3.metric("Accuracy", f"{top_row['accuracy']:.3f}")
    c4.metric("F1", f"{top_row['f1']:.3f}")
    c5.metric("Precision / Recall", f"{top_row['precision']:.3f} / {top_row['recall']:.3f}")

# -----------------------
# TABs
# -----------------------
tab1, tab2, tab3 = st.tabs(["📋 Tabella", "📈 Grafici", "⚖️ Confronto"])

# ===== TAB 1 =====
with tab1:
    st.subheader("Tabella metriche (con arricchimento)")
    styled = (
        df[["model", "accuracy", "precision", "recall", "f1", "score_composito", "rank_score", "rank_"+sort_by]]
        .style
        .format({
            "accuracy": "{:.4f}", "precision": "{:.4f}", "recall": "{:.4f}",
            "f1": "{:.4f}", "score_composito": "{:.4f}"
        })
        .highlight_max(subset=["accuracy", "precision", "recall", "f1", "score_composito"], color="#eafaea")
    )
    st.dataframe(styled, use_container_width=True)

    out = df.copy()
    st.download_button(
        label="⬇️ Scarica CSV arricchito",
        data=out.to_csv(index=False).encode("utf-8"),
        file_name="metrics_enriched.csv",
        mime="text/csv",
        use_container_width=True
    )

# ===== TAB 2 =====
with tab2:
    st.subheader("Distribuzione metriche per modello")
    chart_data = df.melt(id_vars=["model"], value_vars=metrics_cols, var_name="metric", value_name="value")
    # Qui il facet per metrica va bene (è una scelta stilistica); se preferisci grouped bar, togli column=...
    bar = (
        alt.Chart(chart_data)
        .mark_bar()
        .encode(
            x=alt.X("model:N", sort="-y", title="Modello"),
            y=alt.Y("value:Q", title="Valore"),
            column=alt.Column("metric:N", title=None)
        )
        .properties(height=300)
        .resolve_scale(y="independent")
    )
    st.altair_chart(bar, use_container_width=True)

    st.divider()
    st.subheader("Radar chart (profilo per modello)")
    radar_metrics = ["accuracy", "precision", "recall", "f1"]
    model_sel = st.selectbox("Seleziona modello per il radar", df["model"].tolist())
    row = df[df["model"] == model_sel].iloc[0]
    radar_df = pd.DataFrame({"metric": radar_metrics, model_sel: [row[m] for m in radar_metrics]})
    fig = px.line_polar(radar_df, r=model_sel, theta="metric", line_close=True, range_r=[0, 1])
    fig.update_traces(fill="toself")
    st.plotly_chart(fig, use_container_width=True)

# ===== TAB 3 – CONFRONTO (con grouped bar chart) =====
with tab3:
    st.subheader("Confronto tra due modelli")

    cols_sel = st.columns(2)
    model_a = cols_sel[0].selectbox("Modello A", df["model"].tolist(), index=0, key="cmp_a")
    model_b = cols_sel[1].selectbox("Modello B", df["model"].tolist(), index=min(1, len(df)-1), key="cmp_b")

    if model_a == model_b:
        st.info("Seleziona due modelli diversi per un confronto significativo.")

    a = df[df["model"] == model_a].iloc[0]
    b = df[df["model"] == model_b].iloc[0]

    cmp_metrics = ["score_composito", "accuracy", "precision", "recall", "f1"]
    labels = {"score_composito": "Score", "accuracy": "Accuracy", "precision": "Precision", "recall": "Recall", "f1": "F1"}

    vals_a = {m: float(a[m]) for m in cmp_metrics}
    vals_b = {m: float(b[m]) for m in cmp_metrics}
    deltas = {m: vals_a[m] - vals_b[m] for m in cmp_metrics}

    # Toggle baseline per Δ%
    st.markdown("**Baseline Δ%** (divisore per la percentuale)")
    base_choice = st.radio(
        "Scegli la baseline per la percentuale",
        options=["Modello B (Δ% = (A−B)/B)", "Modello A (Δ% = (A−B)/A)"],
        index=0,
        horizontal=True,
        label_visibility="collapsed"
    )
    use_b_as_baseline = base_choice.startswith("Modello B")
    def pct(delta, denom):
        return (delta / denom) if denom != 0 else float("inf")

    pct_deltas = {m: pct(deltas[m], vals_b[m] if use_b_as_baseline else vals_a[m]) for m in cmp_metrics}
    st.caption(f"🔎 Δ% calcolato rispetto a **{model_b if use_b_as_baseline else model_a}**")

    # Riassunto "chi vince"
    wins_a = sum(deltas[m] > 0 for m in cmp_metrics)
    wins_b = sum(deltas[m] < 0 for m in cmp_metrics)
    st.caption(f"🏁 **Vittorie metriche** — {model_a}: {wins_a} | {model_b}: {wins_b} | Pareggi: {len(cmp_metrics) - wins_a - wins_b}")

    # Layout a 3 colonne: A | Δ | B
    cA, cD, cB = st.columns([1.2, 1.0, 1.2])
    cA.markdown(f"### {model_a}")
    cD.markdown("### Δ (A − B)")
    cB.markdown(f"### {model_b}")

    def metric_card(col, title, value):
        col.metric(title, f"{value:.3f}")

    def delta_card(col, title, d_abs, d_pct):
        if np.isfinite(d_pct):
            col.metric(title, f"{d_abs:+.3f}", delta=f"{d_pct:+.1%}")
        else:
            col.metric(title, f"{d_abs:+.3f}", delta="n/a")

    for m in cmp_metrics:
        metric_card(cA, labels[m], vals_a[m])
        delta_card(cD, labels[m], deltas[m], pct_deltas[m])
        metric_card(cB, labels[m], vals_b[m])

    st.divider()

    # Tabella confronto con Δ assoluto e Δ% + evidenziazione
    comp_rows = []
    for m in cmp_metrics:
        winner = model_a if deltas[m] > 0 else (model_b if deltas[m] < 0 else "—")
        comp_rows.append({
            "Metrica": labels[m],
            f"{model_a}": round(vals_a[m], 6),
            f"{model_b}": round(vals_b[m], 6),
            "Δ Assoluto (A−B)": round(deltas[m], 6),
            "Δ %": (f"{pct_deltas[m]*100:.2f}%" if np.isfinite(pct_deltas[m]) else "n/a"),
            "Vince": winner
        })
    comp_df = pd.DataFrame(comp_rows)

    def highlight_delta(val):
        try:
            v = float(str(val).replace("%", ""))
        except Exception:
            return ""
        if isinstance(val, str) and "%" in val:
            return "background-color: #eafaea" if v > 0 else ("background-color: #ffecec" if v < 0 else "")
        return "background-color: #eafaea" if v > 0 else ("background-color: #ffecec" if v < 0 else "")

    styled_comp = (
        comp_df.style
        .format({f"{model_a}":"{:.4f}", f"{model_b}":"{:.4f}", "Δ Assoluto (A−B)":"{:+.4f}"})
        .applymap(highlight_delta, subset=["Δ Assoluto (A−B)", "Δ %"])
    )
    st.subheader("Tabella di confronto (Δ assoluto e Δ%)")
    st.dataframe(styled_comp, use_container_width=True)

    # -------- Grouped Bar Chart (FIX) --------
    st.subheader("Confronto grafico (barre)")
    chart_df = pd.DataFrame({
        "metric": [labels[m] for m in cmp_metrics for _ in (0, 1)],
        "model": [model_a, model_b] * len(cmp_metrics),
        "value": sum(([vals_a[m], vals_b[m]] for m in cmp_metrics), [])
    })
    bar = (
        alt.Chart(chart_df)
        .mark_bar()
        .encode(
            x=alt.X("metric:N", title="Metrica"),
            y=alt.Y("value:Q", title="Valore", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("model:N", title="Modello"),
            # niente facet/column: grouped bar in un singolo grafico
        )
        .properties(height=350)
    )
    st.altair_chart(bar, use_container_width=True)

    # Grafico lineare comparativo
    st.subheader("Confronto grafico (linee)")
    order = ["Score", "Accuracy", "Precision", "Recall", "F1"]
    line_df = pd.DataFrame({
        "metric": order * 2,
        "model": [model_a] * len(order) + [model_b] * len(order),
        "value": [vals_a["score_composito"], vals_a["accuracy"], vals_a["precision"], vals_a["recall"], vals_a["f1"],
                  vals_b["score_composito"], vals_b["accuracy"], vals_b["precision"], vals_b["recall"], vals_b["f1"]]
    })
    line = (
        alt.Chart(line_df)
        .mark_line(point=True)
        .encode(
            x=alt.X("metric:N", sort=order, title="Metrica"),
            y=alt.Y("value:Q", title="Valore", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("model:N", title="Modello")
        )
        .properties(height=280)
    )
    st.altair_chart(line, use_container_width=True)

# -----------------------
# NOTE FINALI
# -----------------------
with st.expander("ℹ️ Note e suggerimenti"):
    st.markdown(
        """
- **Score composito**: combina le metriche normalizzate con i pesi scelti nella sidebar (default: uniformi).
- **Radar**: utile per vedere la “forma” del modello (precision vs recall, ecc).
- **Download**: salva il CSV arricchito con score e ranking per report o versionamento.
- **Consigli**:
  - Aggiungi un selettore di soglia di decisione se lavori con probabilità e vuoi simulare _precision–recall trade-off_.
  - Se hai cross-validation per modello, estendi il CSV con `std` e mostra barre d’errore.
  - Integra metriche aggiuntive (AUC, MCC) se disponibili.
        """
    )
