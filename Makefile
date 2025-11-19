.PHONY: build clean-db run rebuild

IMAGE_NAME ?= openrsvp
CONTAINER_NAME ?= orac

build:
	docker build -t $(IMAGE_NAME) .

clean-db:
	rm -f data/openrsvp.db

run:
	mkdir -p data
	docker run --rm -d --name $(CONTAINER_NAME) -p 8000:8000 -v "./data:/app/data" $(IMAGE_NAME)

seed:
	docker exec $(CONTAINER_NAME) $(IMAGE_NAME) seed-data

rebuild: build clean-db run
