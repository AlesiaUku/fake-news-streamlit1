import streamlit as st
import joblib

# Ngarko modelin dhe vectorizerin
model = joblib.load("pac_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# UI e thjeshtë dhe e bukur
st.set_page_config(page_title="Detektues Lajmesh të Rreme", layout="centered")
st.title("🎓 Universiteti Politeknik i Tiranës")
st.header("📰 Detektimi Automatik i Lajmeve të Rreme")

st.markdown("""
Ky aplikacion përdor algoritmin **Passive Aggressive Classifier** për të analizuar lajmet dhe për të përcaktuar nëse ato janë **të vërteta** apo **të rreme**, bazuar në përmbajtjen tekstuale të tyre.
""")

# Input nga përdoruesi
news = st.text_area("✍️ Shkruaj ose ngjit një lajm për verifikim:")

if st.button("🔍 Verifiko lajmin"):
    if not news.strip():
        st.warning("Ju lutem shkruani një lajm për të vazhduar.")
    else:
        # Transformo dhe parashiko
        x_input = vectorizer.transform([news])
        prediction = model.predict(x_input)

        # Shfaq rezultatin
        if prediction[0] == "FAKE":
            st.error("🚨 Ky është një LAJM I RREMË!")
        else:
            st.success("✅ Ky është një LAJM I VËRTETË!")

# Footer
st.markdown("---")
st.caption("Punuar si pjesë e temës së diplomës në Fakultetin e Teknologjisë së Informacionit – UPT.")

