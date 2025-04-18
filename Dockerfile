FROM python:3.12-slim

WORKDIR /app

# installation de uv
RUN pip install uv

# Création et activation de l'environnement virtuel
RUN python -m venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"

# copie des fichiers de configuration
COPY pyproject.toml .
# copie du fichier streamlit_app.py
COPY api/streamlit_app.py .

# installation des dépendances avec uv (s'exécutera dans l'environnement virtuel)
RUN uv pip install .

# copie des fichiers sources
COPY model/ model/
COPY src/ src/
COPY api/ api
COPY notebook/ notebook


EXPOSE 8501

CMD ["streamlit", "run", "streamlit_app.py"]