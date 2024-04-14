# Set Docker registry and image name for easier reuse
CONTAINER_REGISTRY = dkopylov
IMAGE_NAME = hw7

# Define a Python virtual environment for dependency isolation
venv:
	python -m venv venv
	source venv/bin/activate

# Train the ML model inside a virtual environment
train: venv
	python train.py

# Build Docker image with MLServer for serving the model
build: venv
	python shoose.py default
	mlserver build . -t $(CONTAINER_REGISTRY)/$(IMAGE_NAME)

build-alt: venv
	python choose.py alt
	mlserver build . -t $(CONTAINER_REGISTRY)/$(IMAGE_NAME)

# Push Docker image to registry
push-image:
	docker push $(CONTAINER_REGISTRY)/$(IMAGE_NAME)

# Create a Kind cluster for local Kubernetes testing
kind-cluster:
	kind create cluster --name seldon-cluster --config kind-cluster.yaml --image=kindest/node:v1.21.2

# Install Ambassador as an API gateway using Helm
ambassador: kind-cluster
	helm repo add datawire https://www.getambassador.io
	helm repo update
	helm upgrade --install ambassador datawire/ambassador \
		--set image.repository=docker.io/datawire/ambassador \
		--set service.type=ClusterIP \
		--set replicaCount=1 \
		--set crds.keep=false \
		--set enableAES=false \
		--create-namespace \
		--namespace ambassador

# Install Seldon Core in Kubernetes to manage ML deployments
seldon-core: kind-cluster
	helm repo add seldon https://storage.googleapis.com/seldon-charts
	helm repo update
	helm upgrade --install seldon-core seldon/seldon-core-operator \
		--set crd.create=true \
		--set usageMetrics.enabled=true \
		--set ambassador.enabled=true \
		--create-namespace \
		--namespace seldon-system

# Install Prometheus for monitoring via Helm
prometheus: kind-cluster
	helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
	helm repo update
	helm upgrade --install seldon-monitoring prometheus-community/kube-prometheus-stack \
		--set fullnameOverride=seldon-monitoring \
		--create-namespace \
		--namespace seldon-monitoring

# Install Grafana for metrics visualization via Helm
grafana: kind-cluster
	helm repo add grafana https://grafana.github.io/helm-charts
	helm repo update
	helm upgrade --install grafana-seldon-monitoring grafana/grafana \
		--set version=6.56.1 \
		--values service.yaml \
		--namespace seldon-monitoring

# Deploy all components
deploy-everything: kind-cluster ambassador seldon-core prometheus grafana

# Deploy the application using Seldon to manage the machine learning lifecycle
deploy:
	kubectl apply -f seldondeployment.yaml
	kubectl apply -f podmonitor.yaml

# Delete the deployed predictor application
delete-deployment:
	kubectl delete -f seldondeployment.yaml

# Port forwarding commands for local access
port-forward-grafana:
	kubectl port-forward svc/grafana-seldon-monitoring 3000:80 --namespace seldon-monitoring

port-forward-prometheus:
	kubectl port-forward svc/seldon-monitoring-prometheus 9090 --namespace seldon-monitoring

# Test the deployment by sending a test request
test-request:
	curl -X POST -H "Content-Type: application/json" \
        -d '{"data": {"ndarray": [[your_features_here]]}}' \
        http://localhost:8000/seldon/default/$(MODEL_NAME)/api/v1.0/predictions

# Display help for available Makefile commands
help:
	@echo "Available commands:"
	@echo "train                   - Train the model."
	@echo "build                   - Build Docker image with MLServer (one)."
    @echo "build-alt               - Build Docker image with MLServer (two)."
	@echo "push-image              - Push Docker image to registry."
	@echo "kind-cluster            - Create a Kind cluster."
	@echo "ambassador              - Install Ambassador."
	@echo "seldon-core             - Install Seldon Core."
	@echo "prometheus              - Install Prometheus."
	@echo "grafana                 - Install Grafana."
	@echo "deploy-everything       - Deploy all infrastructure components."
	@echo "deploy                  - Deploy the application."
	@echo "delete-deployment       - Delete the application deployment."
	@echo "port-forward-grafana    - Port-forward for Grafana."
	@echo "port-forward-prometheus - Port-forward for Prometheus."
	@echo "test-request            - Send a test request."
	@echo "help                    - Display this help message."
