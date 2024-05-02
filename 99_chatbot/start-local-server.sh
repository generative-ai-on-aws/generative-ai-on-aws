docker pull ghcr.io/huggingface/text-generation-inference:latest --platform linux/amd64

docker run --platform linux/amd64 --shm-size 1g -p 8081:80 -v $PWD/data:/data ghcr.io/huggingface/text-generation-inference:latest --model-id mistralai/Mixtral-8x7B-Instruct-v0.1 
