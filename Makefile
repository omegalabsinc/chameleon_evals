default: chameleon-evals

sh:
	docker run -it --rm \
		--ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --gpus=all \
		--cap-add SYS_PTRACE --cap-add=SYS_ADMIN --ulimit core=0 \
		-v $(shell pwd):/app \
		chameleon-evals

chameleon-evals:
	docker build -t $@ -f Dockerfile .

