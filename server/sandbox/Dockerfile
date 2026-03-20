FROM python:3.13-slim

RUN pip install --no-cache-dir numpy

COPY runner.py /app/runner.py

WORKDIR /app

# engine/ and agents/ are mounted at runtime (read-only)
# /app/output/ is mounted for writing results

ENTRYPOINT ["python", "-u", "runner.py"]
