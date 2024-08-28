FROM python:3.10

RUN pip install poetry

RUN apt-get update && apt-get install -y build-essential libpq-dev

ENV APP_HOME /app

WORKDIR $APP_HOME

COPY pyproject.toml poetry.lock ./

RUN poetry config virtualenvs.create false && poetry install --no-root

COPY . .

EXPOSE 8000

CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]

