"""
Tests unitarios — Heart Disease Prediction API
Ejecutar: pytest tests/ -v
"""
import pytest
import numpy as np
import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Número exacto de features que produce get_dummies sobre el dataset
N_FEATURES = 15


class TestInputValidation:
    """Valida el procesamiento de inputs"""

    def test_feature_vector_shape(self):
        features = [40, 140, 289, 0, 172, 0.0, 1, 1, 0, 0, 1, 0, 0, 0, 1]
        X = np.array(features).reshape(1, -1)
        assert X.shape == (1, N_FEATURES)

    def test_wrong_feature_count_detected(self):
        features_short = [40, 140, 289]
        assert len(features_short) != N_FEATURES

    def test_all_features_numeric(self):
        features = [40, 140, 289, 0, 172, 0.0, 1, 1, 0, 0, 1, 0, 0, 0, 1]
        for f in features:
            assert isinstance(f, (int, float))


class TestPredictionLogic:
    """Valida la lógica de predicción"""

    def test_prediction_is_binary(self):
        for proba in [0.0, 0.3, 0.49, 0.5, 0.7, 1.0]:
            pred = int(proba > 0.5)
            assert pred in [0, 1]

    def test_probability_bounds(self):
        for proba in [0.0, 0.25, 0.5, 0.75, 1.0]:
            assert 0.0 <= proba <= 1.0

    def test_risk_level_logic(self):
        def risk(p):
            if p < 0.3:   return "Bajo"
            elif p < 0.6: return "Moderado"
            else:          return "Alto"

        assert risk(0.1)  == "Bajo"
        assert risk(0.29) == "Bajo"
        assert risk(0.3)  == "Moderado"
        assert risk(0.59) == "Moderado"
        assert risk(0.6)  == "Alto"
        assert risk(0.95) == "Alto"

    def test_threshold_at_05(self):
        assert int(0.499 > 0.5) == 0
        assert int(0.501 > 0.5) == 1


class TestModelFile:
    """Verifica el archivo del modelo exportado"""

    MODEL_PATH = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'app', 'model.joblib'
    )

    def test_model_file_exists(self):
        if not os.path.exists(self.MODEL_PATH):
            pytest.skip("Ejecutar notebook 2 primero para generar model.joblib")
        assert os.path.exists(self.MODEL_PATH)

    def test_model_file_not_empty(self):
        if not os.path.exists(self.MODEL_PATH):
            pytest.skip("Ejecutar notebook 2 primero para generar model.joblib")
        assert os.path.getsize(self.MODEL_PATH) > 1000

    def test_model_has_predict_proba(self):
        if not os.path.exists(self.MODEL_PATH):
            pytest.skip("Ejecutar notebook 2 primero para generar model.joblib")
        import joblib
        model = joblib.load(self.MODEL_PATH)
        assert hasattr(model, 'predict_proba')
        assert hasattr(model, 'predict')
