#!/usr/bin/env bash
# =============================================================
# build_jbook.sh — Construye el Jupyter Book del proyecto
# Uso: bash build_jbook.sh
# =============================================================

set -e

echo "=============================================="
echo "  Heart Disease MLOps — Jupyter Book Builder  "
echo "=============================================="
echo ""

# Detectar Python (prioriza Python 3.11)
PYTHON_BIN="python"

if command -v py &> /dev/null; then
    PYTHON_BIN="py -3.11"
fi

echo "🐍 Usando Python: $PYTHON_BIN"

# 1. Instalar jupyter-book versión estable
echo ""
echo "📦 Instalando/verificando jupyter-book..."
$PYTHON_BIN -m pip install jupyter-book==0.15.1 --quiet

# 2. Limpiar build anterior
echo ""
echo "🧹 Limpiando build anterior..."
rm -rf _build

# 3. Construir el libro (forzando módulo correcto)
echo ""
echo "📚 Construyendo Jupyter Book..."
$PYTHON_BIN -m jupyter_book.cli.main build . -v

# 4. Resultado
echo ""
echo "=============================================="
echo "  ✅ Jupyter Book generado correctamente"
echo "=============================================="
echo ""
echo "  Abre el libro en tu navegador:"
echo "  👉  _build/html/index.html"
echo ""

# 5. Abrir automáticamente (multiplataforma)
if command -v open &> /dev/null; then
    open _build/html/index.html
elif command -v xdg-open &> /dev/null; then
    xdg-open _build/html/index.html
elif command -v start &> /dev/null; then
    start _build/html/index.html
else
    echo "⚠️ No se pudo abrir automáticamente. Ábrelo manualmente."
fi