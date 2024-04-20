.PHONY: all

train:
	python train.py

build:
	mlserver build . -t mirckos/uplift-model:v1.0
	docker push mirckos/uplift-model:v1.0

kind-cluster:
	docker pull kindest/node:v1.20.7
	sudo kind create cluster --name seldon-cluster --config kind-cluster.yaml --image=kindest/node:v1.20.7

ambassador:
	helm repo add datawire https://www.getambassador.io
	helm repo update
	helm upgrade --install ambassador datawire/ambassador --set image.repository=docker.io/datawire/ambassador --set service.type=ClusterIP --set replicaCount=1 --set crds.keep=false --set enableAES=false --create-namespace --namespace ambassador

seldon-core:
	helm install seldon-core seldon-core-operator --repo https://storage.googleapis.com/seldon-charts --set usageMetrics.enabled=true --set ambassador.enabled=true --create-namespace --namespace seldon-system --version 1.15

prometheus:
	helm upgrade --install seldon-monitoring kube-prometheus --version 8.9.1 --set fullnameOverride=seldon-monitoring --create-namespace --namespace seldon-monitoring --repo https://charts.bitnami.com/bitnami

grafana:
	helm repo add grafana https://grafana.github.io/helm-charts
	helm upgrade --install grafana-seldon-monitoring grafana/grafana --version 6.56.1 --values values_grafana_local.yaml --namespace seldon-monitoring

deploy-model:
	kubectl apply -f seldon-deployment.yaml

port-forward:
	kubectl port-forward -n seldon-monitoring svc/grafana-seldon-monitoring 3000:80 &
	kubectl port-forward svc/ambassador 8003:8003 &
	kubectl port-forward svc/prometheus -n monitoring 9090:9090 &
	kubectl port-forward svc/uplift-model-uplift-model-uplift-model 9000:9000 &


test:
	curl -X POST http://localhost:8003/seldon/default/uplift-model/api/v1.0/predictions -d '{"data": {"x": [[1,1,1], [1,2,3]}}'


all: train kind-cluster ambassador seldon-core prometheus grafana deploy-model port-forward