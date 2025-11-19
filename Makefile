.PHONY: build clean-db run rebuild

build:
	docker build -t openrsvp .

clean-db:
	rm -f data/openrsvp.db

run:
	mkdir -p data
	docker run --rm -p 8000:8000 -v "./data:/app/data" openrsvp

rebuild: build clean-db run
