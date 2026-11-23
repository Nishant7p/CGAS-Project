# FlavorPrint (D5) for Hugging Face Spaces: Docker SDK with Streamlit (the Streamlit SDK was deprecated in 2025).
# The Space repo is assembled by app/package_space.py: this Dockerfile, streamlit_app.py, requirements.txt,
# artifacts/ (CC BY-NC-SA 4.0) and the flavoraudit source (MIT).
FROM python:3.12-slim

RUN useradd -m -u 1000 user
WORKDIR /home/user/app
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY --chown=user flavoraudit ./flavoraudit_src/flavoraudit
COPY --chown=user streamlit_app.py .
COPY --chown=user .streamlit ./.streamlit
COPY --chown=user artifacts ./artifacts
ENV PYTHONPATH=/home/user/app/flavoraudit_src \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
USER user
EXPOSE 8501
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
