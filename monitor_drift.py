"""
Etapa 6 — Monitoreo de Deriva de Datos con Evidently
Genera drift_report.html comparando train vs test.

Instalar: pip install evidently
Ejecutar: python monitor_drift.py
"""
import pandas as pd
from sklearn.model_selection import train_test_split


def generate_drift_report():
    try:
        from evidently.report import Report
        from evidently.metric_preset import DataDriftPreset, DataQualityPreset
    except ImportError:
        print("Instala evidently primero:  pip install evidently")
        return

    print("Cargando y preparando datos...")
    df = pd.read_csv("heart.csv")
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    df_enc = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    X = df_enc.drop("HeartDisease", axis=1)
    y = df_enc["HeartDisease"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    train_data = X_train.copy()
    train_data["HeartDisease"] = y_train.values

    test_data = X_test.copy()
    test_data["HeartDisease"] = y_test.values

    print("Generando reporte de Data Drift y Data Quality...")
    report = Report(metrics=[DataDriftPreset(), DataQualityPreset()])
    report.run(reference_data=train_data, current_data=test_data)
    report.save_html("drift_report.html")

    print("✓ Reporte guardado: drift_report.html")
    print("  Ábrelo en tu navegador para ver los resultados.")


if __name__ == "__main__":
    generate_drift_report()
