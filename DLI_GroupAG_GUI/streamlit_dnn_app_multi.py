
# streamlit_dnn_app_multi.py
# 2-page Streamlit App for Phishing Detection with DNN (with session_state fix)
#
# Run:
#   pip install streamlit tensorflow scikit-learn pandas numpy matplotlib
#   streamlit run streamlit_dnn_app_multi.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

from urllib.parse import urlparse
import whois
import socket
import datetime

def extract_features_from_url(url: str, feature_columns: list):
    parsed = urlparse(url)
    hostname = parsed.netloc
    path = parsed.path

    features = {}

    # ---------------- Basic string features ----------------
    features["length_url"] = len(url)
    features["length_hostname"] = len(hostname)
    features["ip"] = 1 if hostname.replace(".", "").isdigit() else 0
    features["nb_dots"] = url.count(".")
    features["nb_hyphens"] = url.count("-")
    features["nb_at"] = url.count("@")
    features["nb_qm"] = url.count("?")
    features["nb_and"] = url.count("&")
    features["nb_or"] = url.count("|")
    features["nb_eq"] = url.count("=")
    features["nb_underscore"] = url.count("_")
    features["nb_tilde"] = url.count("~")
    features["nb_percent"] = url.count("%")
    features["nb_slash"] = url.count("/")
    features["nb_star"] = url.count("*")
    features["nb_colon"] = url.count(":")
    features["nb_comma"] = url.count(",")
    features["nb_semicolumn"] = url.count(";")
    features["nb_dollar"] = url.count("$")
    features["nb_space"] = url.count(" ")
    features["nb_www"] = url.lower().count("www")
    features["nb_com"] = url.lower().count(".com")
    features["nb_dslash"] = url.count("//")
    features["http_in_path"] = 1 if "http" in path else 0
    features["https_token"] = 1 if "https" in url[8:] else 0
    features["ratio_digits_url"] = sum(c.isdigit() for c in url) / len(url)
    features["ratio_digits_host"] = sum(c.isdigit() for c in hostname) / max(1, len(hostname))

    # ---------------- WHOIS features ----------------
    try:
        domain_info = whois.whois(hostname)
        if domain_info:
            features["whois_registered_domain"] = 1
            if isinstance(domain_info.creation_date, list):
                creation_date = domain_info.creation_date[0]
            else:
                creation_date = domain_info.creation_date
            if isinstance(domain_info.expiration_date, list):
                expiration_date = domain_info.expiration_date[0]
            else:
                expiration_date = domain_info.expiration_date

            if creation_date and expiration_date:
                features["domain_registration_length"] = (expiration_date - creation_date).days
            else:
                features["domain_registration_length"] = 0
            if creation_date:
                features["domain_age"] = (datetime.datetime.now() - creation_date).days
            else:
                features["domain_age"] = 0
        else:
            features["whois_registered_domain"] = 0
            features["domain_registration_length"] = 0
            features["domain_age"] = 0
    except Exception:
        features["whois_registered_domain"] = 0
        features["domain_registration_length"] = 0
        features["domain_age"] = 0

    # ---------------- DNS record ----------------
    try:
        socket.gethostbyname(hostname)
        features["dns_record"] = 1
    except Exception:
        features["dns_record"] = 0

    # ---------------- Placeholder web traffic ----------------
    features["web_traffic"] = 0

    # ---------------- Fill missing ----------------
    for col in feature_columns:
        if col not in features:
            features[col] = 0

    return pd.DataFrame([features])[feature_columns]


st.set_page_config(page_title="Phishing Detection – DNN", layout="wide")

def _prepare_dataframe(df: pd.DataFrame):
    if "status" in df.columns:
        df = df.copy()
        df["Label"] = df["status"].map({"legitimate": 0, "phishing": 1})
        drop_cols = [c for c in ["url", "status"] if c in df.columns]
        df = df.drop(columns=drop_cols)
    if "Label" not in df.columns:
        raise ValueError("Dataset must include 'status' (legitimate/phishing) or 'Label' (0/1).")
    X = df.drop(columns=["Label"]).values
    y = df["Label"].values.astype(int)
    return df, X, y

# ---------------------- Page Navigation ----------------------
page = st.sidebar.selectbox("Choose a page:", ["1️⃣ Train & Evaluate DNN", "2️⃣ Detect URL"])

# ---------------------- Page 1 ----------------------
if page == "1️⃣ Train & Evaluate DNN":
    st.title("🧠 Train & Evaluate DNN Model")
    train_file = st.file_uploader("Upload training CSV", type=["csv"], key="train")
    if train_file is not None:
        df = pd.read_csv(train_file)
        try:
            df, X, y = _prepare_dataframe(df)
            FEATURE_COLUMNS = df.drop(columns=["Label"]).columns.tolist()

            SCALER = StandardScaler()
            X_scaled = SCALER.fit_transform(X)
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

            MODEL = Sequential([
                Dense(128, activation="relu", input_shape=(X_train.shape[1],)),
                Dropout(0.3),
                Dense(64, activation="relu"),
                Dropout(0.3),
                Dense(1, activation="sigmoid")
            ])
            MODEL.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

            with st.spinner("Training model..."):
                HISTORY = MODEL.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test), verbose=0)

            # Evaluate
            loss, acc = MODEL.evaluate(X_test, y_test, verbose=0)
            y_prob = MODEL.predict(X_test, verbose=0).ravel()
            y_pred = (y_prob > 0.5).astype(int)
            auc = roc_auc_score(y_test, y_prob)

            st.success("✅ Model trained and evaluated!")
            st.metric("Accuracy", f"{acc:.4f}")
            st.metric("ROC AUC", f"{auc:.4f}")
            st.metric("Loss", f"{loss:.4f}")

            # Classification report
            st.text("Classification Report:\n" + classification_report(y_test, y_pred, target_names=["Legitimate", "Phishing"]))

            # Confusion matrix plot
            cm = confusion_matrix(y_test, y_pred)
            fig_cm, ax_cm = plt.subplots()
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Legitimate", "Phishing"])
            disp.plot(cmap=plt.cm.Blues, ax=ax_cm)
            st.pyplot(fig_cm)

            # Accuracy and loss curves
            fig_curves, ax_curves = plt.subplots(1,2, figsize=(10,4))
            ax_curves[0].plot(HISTORY.history["accuracy"], label="Train Acc")
            ax_curves[0].plot(HISTORY.history["val_accuracy"], label="Val Acc")
            ax_curves[0].set_title("Accuracy Over Epochs")
            ax_curves[0].legend()
            ax_curves[1].plot(HISTORY.history["loss"], label="Train Loss")
            ax_curves[1].plot(HISTORY.history["val_loss"], label="Val Loss")
            ax_curves[1].set_title("Loss Over Epochs")
            ax_curves[1].legend()
            st.pyplot(fig_curves)

            # Save objects in session_state
            st.session_state.MODEL = MODEL
            st.session_state.SCALER = SCALER
            st.session_state.FEATURE_COLUMNS = FEATURE_COLUMNS
            st.session_state.HISTORY = HISTORY

        except Exception as e:
            st.error(f"Error during training: {e}")

# ---------------------- Page 2 ----------------------
elif page == "2️⃣ Detect URL":
    st.title("🔗 Detect Phishing URL")
    st.write("Enter a URL to check if it's phishing or legitimate.")

    url_input = st.text_input("Enter URL:")
    if st.button("Check URL"):
        if "MODEL" not in st.session_state or "SCALER" not in st.session_state or "FEATURE_COLUMNS" not in st.session_state:
            st.warning("⚠️ Please train the DNN first on Page 1.")
        else:
            MODEL = st.session_state.MODEL
            SCALER = st.session_state.SCALER
            FEATURE_COLUMNS = st.session_state.FEATURE_COLUMNS

            # Use full extractor
            df_features = extract_features_from_url(url_input, FEATURE_COLUMNS)

            X_scaled = SCALER.transform(df_features.values)
            prob = MODEL.predict(X_scaled, verbose=0).ravel()[0]
            pred = "Phishing" if prob > 0.5 else "Legitimate"

            st.write(f"Prediction: **{pred}** (prob={prob:.4f})")
