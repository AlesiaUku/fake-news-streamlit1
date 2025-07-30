import streamlit as st
import joblib

# Ngarko modelin Passive Aggressive dhe vectorizer-in
model = joblib.load("pac_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Konfigurimi i faqes
st.set_page_config(page_title="Detektues Lajmesh", layout="centered")
st.title("🎓 Universiteti Politeknik i Tiranës")
st.header("📘 Detektues i Lajmeve të Rreme")

st.markdown("""
Ky aplikacion përdor algoritmin **Passive Aggressive Classifier** për të zbuluar nëse një lajm është i **vërtetë** apo i **rremë**, në bazë të përmbajtjes së tij.
""")

# Fusha për të shtypur lajmin
news = st.text_area("📰 Shkruani lajmin që dëshironi të verifikoni:")

# Butoni për verifikim
if st.button("🔍 Verifiko lajmin"):
    if news.strip() == "":
        st.warning("Ju lutem shkruani një lajm për ta analizuar.")
    else:
        x = vectorizer.transform([news])
        prediction = model.predict(x)

        if prediction[0] == "FAKE":
            st.error("🎉 Ky është një LAJM I RREMË!")
        else:
            st.success("✅ Ky është një LAJM I VËRTETË!")

# Footer
st.markdown("---")
st.caption("Punuar si pjesë e temës së diplomës në Fakultetin e Teknologjisë së Informacionit.")
