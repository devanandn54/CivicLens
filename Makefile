.PHONY: schema up down api evals

up:
	docker compose up -d

down:
	docker compose down

schema:
	docker exec -i civiclens-db psql -U civiclens -d civiclens < db/schema.sql

api:
	cd api && uvicorn app.main:app --reload

evals:
	python evals/run_evals.py --k 5
