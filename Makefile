run:
	poetry run streamlit run app.py

build:
	@if [ -z "$(IMAGE_VERSION)" ]; then \
		echo "❌ ERROR: IMAGE_VERSION is empty. Please set it."; \
		echo "   Example: make build IMAGE_VERSION=0.0.1"; \
		exit 1; \
	fi
	docker buildx build \
	--platform linux/arm64 \
	-t chihyuyeh/python-data-analytics:$(IMAGE_VERSION) \
	--load \
	.	

build-and-push:
	@if [ -z "$(IMAGE_VERSION)" ]; then \
		echo "❌ ERROR: IMAGE_VERSION is empty. Please set it."; \
		echo "   Example: make build IMAGE_VERSION=0.0.1"; \
		exit 1; \
	fi
	docker buildx build \
		--platform linux/amd64,linux/arm64 \
		-t chihyuyeh/python-data-analytics:$(IMAGE_VERSION) \
		--push \
		.