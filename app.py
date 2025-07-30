import streamlit as st
import joblib

# Ngarkimi i modelit dhe vectorizer-it
model = joblib.load("rf_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Header personalizuar me emrin e universitetit
st.image("https://upload.wikimedia.org/wikipedia/sq/f/fc/Logo_-_Universiteti_Politeknik_i_Tiranës.png", width=100)
st.title("🎓 Universiteti Politeknik i Tiranës")
st.subheader("📘 Detektues i Lajmeve të Rreme")
st.markdown("""
Ky aplikacion përdor algoritmin **Random Forest** për të zbuluar nëse një lajm është i **vërtetë** apo **i rremë**, në bazë të analizës së përmbajtjes së tij.
""")

# Input nga përdoruesi
user_input = st.text_area("📝 Shkruani lajmin që dëshironi të verifikoni:")

# Butoni për analizë
if st.button("🔍 Verifiko lajmin"):
    if user_input.strip() == "":
        st.warning("Ju lutem shkruani një lajm për të vazhduar.")
    else:
        vec = vectorizer.transform([user_input])
        pred = model.predict(vec)[0]

        if pred == 1:
            st.success("✅ Ky është një LAJM I VËRTETË!")
        else:
            st.error("🚨 Ky është një LAJM I RREMË!")

# Footer
st.markdown("---")
st.caption("Punuar si pjesë e temës së diplomës në Fakultetin e Teknologjisë së Informacionit.")
