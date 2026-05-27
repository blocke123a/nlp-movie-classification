import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from streamlit_pipeline import load_models, predict, GENRES

# ── page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title='Movie Genre Predictor',
    page_icon='🎬',
    layout='wide'
)

# ── load models once and cache ───────────────────────────────────────────────
@st.cache_resource
def get_models():
    return load_models()

models, preprocessors = get_models()

MODEL_LABELS = {
    'naive_bayes': 'Naive Bayes',
    'logistic': 'Logistic Regression',
    'lstm': 'LSTM',
    'transformer': 'Transformer',
}

# ── header ───────────────────────────────────────────────────────────────────
st.title('Movie Genre Predictor')
st.markdown(
    'Enter a real or made-up movie summary and see what genre our models predict.'
)
st.divider()

# input 
summary = st.text_area(
    'Movie summary',
    height=180,
    placeholder='e.g. A young wizard discovers he has magical powers and enrolls in a school for witches and wizards...'
)

selected_models = st.multiselect(
    'Models to run',
    options=list(MODEL_LABELS.keys()),
    default=list(MODEL_LABELS.keys()),
    format_func=lambda k: MODEL_LABELS[k]
)

run = st.button('Predict genre', type='primary', disabled=not summary or not selected_models)

# prediction 
if run:
    st.divider()

    results = {}
    with st.spinner('Running models...'):
        for model_name in selected_models:
            results[model_name] = predict(summary, model_name, models, preprocessors)

    # cleaned text expander
    with st.expander('See cleaned text'):
        st.write(list(results.values())[0]['cleaned_text'])

    # results: one column per model, vertical bar charts
    cols = st.columns(len(results))

    for col, (model_name, result) in zip(cols, results.items()):
        with col:
            st.subheader(MODEL_LABELS[model_name])

            probs = result['probabilities']
            top_genre = result['genre']
            colors = ['green' if g == top_genre else 'steelblue' for g in probs.keys()]

            fig, ax = plt.subplots(figsize=(3.2, 3.5))
            bars = ax.bar(
                list(probs.keys()),
                list(probs.values()),
                color=colors
            )
            ax.set_ylim(0, 1)
            ax.set_ylabel('Probability')
            ax.bar_label(bars, fmt='%.2f', padding=3, fontsize=9)
            ax.tick_params(axis='x', labelsize=9, rotation=15)
            ax.tick_params(axis='y', labelsize=9)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

    st.divider()

    # ── agreement summary ────────────────────────────────────────────────────
    if len(results) > 1:
        predictions = [r['genre'] for r in results.values()]
        all_agree = len(set(predictions)) == 1

        if all_agree:
            st.success(f'✅ All models agree: **{predictions[0]}**')
        else:
            summary_df = pd.DataFrame({
                'Model': [MODEL_LABELS[m] for m in results],
                'Prediction': predictions,
            })
            st.warning('Models disagree:')
            st.dataframe(summary_df, hide_index=True, use_container_width=True)