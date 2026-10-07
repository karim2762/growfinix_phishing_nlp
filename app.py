"""
app.py - Streamlit demo.   Run:  streamlit run app.py

EDUCATIONAL DEMO ONLY. See the disclaimer in the app and in README.md.
"""
import streamlit as st

from src.predict import MODELS, Detector, highlight_html

st.set_page_config(page_title="Phishing Detector (demo)", page_icon="🎣")
st.title("🎣 Phishing / Spam Detector")
st.caption("A bidirectional LSTM trained on public data. Paste an email, SMS or link.")

st.warning(
    "**Educational project - not a security tool.** The model can be wrong in both "
    "directions (missed phishing, false alarms) and is trained on a limited public dataset. "
    "Never rely on it to decide whether something is safe. Don't paste confidential or "
    "personal information. If in doubt, don't click - contact the sender through a "
    "channel you already trust."
)


@st.cache_resource  # load the model once, not on every button click
def get_detector() -> Detector:
    return Detector()


if not (MODELS / "lstm.pt").exists():
    st.error("No trained model found. Run `python download_data.py` and then `python train.py` first.")
    st.stop()

detector = get_detector()
text = st.text_area("Email text or URL", height=200, placeholder="Paste here...")

if st.button("Analyze", type="primary") and text.strip():
    res = detector.predict(text)
    prob = res["probability"]

    if res["is_phishing"]:
        st.error(f"🚨 **Phishing / Spam** - probability {prob:.1%}")
    else:
        st.success(f"✅ **Safe** - phishing probability {prob:.1%}")
    st.progress(min(max(prob, 0.0), 1.0))
    st.caption(f"Decision threshold: {detector.threshold:.2f} (tuned to favour catching phishing). "
               f"Contains a link: {'yes' if res['has_url'] else 'no'}.")

    scores = detector.suspicious_words(text)
    st.subheader("Suspicious words")
    if scores:
        st.markdown(highlight_html(text, scores), unsafe_allow_html=True)  # text is HTML-escaped inside
        st.caption("Darker red = removing that word lowers the phishing score more.")
    else:
        st.write("No single word stood out.")
