import streamlit as st
import pandas as pd
import numpy as np
import joblib  # Untuk load model
from sklearn.preprocessing import StandardScaler

# **Load model dan scaler**
model_data = joblib.load("model.pkl")  # Load model
scaler = joblib.load("scaler.pkl")  # Load scaler
model = model_data["svm"] # krn accuracy paling tinggi

# **Judul Aplikasi**
st.title("🎓 Student Academic Performance 📚")

st.markdown("Masukkan nilai untuk memprediksi apakah siswa akan **PASS** or **FAIL**.")

# **Sidebar untuk input data**
st.sidebar.header("📝 Input Data ")

def user_input():
    G1 = st.sidebar.slider("📊 Nilai G1", 0, 20, 10)
    G2 = st.sidebar.slider("📊 Nilai G2", 0, 20, 10)
    study_time = st.sidebar.slider("⏳ Waktu Belajar (jam/minggu)", 1, 10, 5)
    absences = st.sidebar.slider("🚶‍♂️ Jumlah Absen", 0, 30, 5)
    failures = st.sidebar.slider("Jumlah Kegagalan", 0, 4, 1)
    health = st.sidebar.slider("Kesehatan (1-5)", 1, 5, 3)
    goout = st.sidebar.slider("Sering Keluar Malam (1-5)", 1, 5, 3)
    Dalc = st.sidebar.slider("Konsumsi Alkohol Harian (1-5)", 1, 5, 3)
    Walc = st.sidebar.slider("Konsumsi Alkohol Akhir Pekan (1-5)", 1, 5, 3)
    Medu = st.sidebar.slider("Pendidikan Ibu (0-4)", 0, 4, 2)
    Fedu = st.sidebar.slider("Pendidikan Ayah (0-4)", 0, 4, 2)

    # **Fitur kategori (one-hot encoding di training harus dipakai di sini juga!)**
    Mjob = st.sidebar.selectbox("Pekerjaan Ibu", ["teacher", "health", "services", "at_home", "other"])
    Fjob = st.sidebar.selectbox("Pekerjaan Ayah", ["teacher", "health", "services", "at_home", "other"])
    reason = st.sidebar.selectbox("Alasan Memilih Sekolah", ["home", "reputation", "course", "other"])
    guardian = st.sidebar.selectbox("Wali", ["mother", "father", "other"])
    
    data = {
        "G1": [G1], "G2": [G2], "studytime": [study_time], "absences": [absences],
        "failures": [failures], "health": [health], "goout": [goout],
        "Dalc": [Dalc], "Walc": [Walc], "Medu": [Medu], "Fedu": [Fedu],
        "Mjob": [Mjob], "Fjob": [Fjob], "reason": [reason], "guardian": [guardian]
    }
    return pd.DataFrame(data)

# Input data dulu tanpa prediksi
df_input = user_input()

# **Tombol Prediksi**
if st.sidebar.button("🔮 Prediksi Kelulusan"):
    # **One-Hot Encoding**
    df_input = pd.get_dummies(df_input)

    # **Pastikan urutan fitur sesuai model**
    original_feature_names = scaler.feature_names_in_
    df_input = df_input.reindex(columns=original_feature_names, fill_value=0)

    # **Normalisasi Data**
    df_input_scaled = scaler.transform(df_input)

    # **Prediksi**
    prediction = model.predict(df_input_scaled)
    prediction_proba = model.predict_proba(df_input_scaled)

    # **Tampilkan Hasil**
    st.subheader("📌 Hasil Prediksi:")

    if prediction[0] == 1:
        st.success("✅ **Mahasiswa Diprediksi LULUS!**")
        st.metric(label="🎯 Status", value="LULUS ✅", delta=f"{prediction_proba[0][1]*100:.2f}%")
    else:
        st.error("❌ **Mahasiswa Diprediksi TIDAK LULUS!**")
        st.metric(label="⚠️ Status", value="TIDAK LULUS ❌", delta=f"{prediction_proba[0][0]*100:.2f}%")

    # **Tampilkan Probabilitas**
    st.subheader("📊 Probabilitas Prediksi:")
    st.progress(int(prediction_proba[0][1] * 100))  # Bar untuk probabilitas kelulusan
    st.write(f"🔥 **Lulus**: {prediction_proba[0][1]*100:.2f}%")
    st.progress(int(prediction_proba[0][0] * 100))  # Bar untuk probabilitas tidak lulus
    st.write(f"❄️ **Tidak Lulus**: {prediction_proba[0][0]*100:.2f}%")