default: chameleon-evals

sh:
	docker run -it --rm \
		--ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --gpus=all \
		--cap-add SYS_PTRACE --cap-add=SYS_ADMIN --ulimit core=0 \
		-v $(shell pwd):/app \
		chameleon-evals

chameleon-evals:
	docker build -t $@ -f Dockerfile .

run_eval_7b:
	docker run -d --rm \
		--ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --gpus=all \
		--cap-add SYS_PTRACE --cap-add=SYS_ADMIN --ulimit core=0 \
		-v $(shell pwd):/app \
		-e HF_TOKEN=$(HF_TOKEN) \
		chameleon-evals \
		python3 recipes/eval.py --config config/7b_eval.yaml
