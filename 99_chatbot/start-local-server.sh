#model1=OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5
#model2=mosaicml/mpt-7b-chat
#model3=microsoft/DialoGPT-medium
#model4=togethercomputer/RedPajama-INCITE-Chat-7B-v0.1

#num_shard=1
#volume=$PWD/data # share a volume with the Docker container to avoid downloading weights every run

docker pull ghcr.io/huggingface/text-generation-inference:latest

nohup docker run --gpus 4 --shm-size 1g -p 8080:80 -v $PWD/data:/data ghcr.io/huggingface/text-generation-inference:latest --model-id h2oai/h2ogpt-4096-llama2-70b-chat --quantize bitsandbytes > nohup-local.out &

#nohup docker run --gpus 4 --shm-size 1g -p 8081:80 -v $PWD/data:/data ghcr.io/huggingface/text-generation-inference:latest --model-id h2oai/h2ogpt-gm-oasst1-en-2048-falcon-40b-v2 --trust-remote-code --quantize bitsandbytes > nohup-1.out &

#nohup docker run --gpus 2 --shm-size 1g -p 8081:80 -v $PWD/data:/data ghcr.io/huggingface/text-generation-inference:latest --model-id openlm-research/open_llama_7b_v2 --quantize bitsandbytes > nohup-2.out &

#nohup docker run --gpus 4 --shm-size 1g -p 8080:80 -v $PWD/data:/data ghcr.io/huggingface/text-generation-inference:latest --model-id TheBloke/OpenAssistant-SFT-7-Llama-30B-HF --num-shard 4 --quantize bitsandbytes > nohup-1.out &

#nohup docker run --gpus 4 --shm-size 1g -p 8081:80 -v $PWD/data:/data ghcr.io/huggingface/text-generation-inference:latest --model-id h2oai/h2ogpt-gm-oasst1-en-2048-falcon-40b-v2 --num-shard 4 --max-batch-prefill-tokens 2048 --trust-remote-code > nohup-1.out &

#nohup docker run --gpus 4 --shm-size 1g -p 8080:80 -v $PWD/data:/data ghcr.io/huggingface/text-generation-inference:latest --model-id mosaicml/mpt-7b --trust-remote-code > nohup-1.out &

#nohup docker run --gpus 4 --shm-size 1g -p 8080:80 -v $PWD/data:/data ghcr.io/huggingface/text-generation-inference:latest --model-id OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5 --num-shard 4 --quantize bitsandbytes > nohup-1.out &

#nohup docker run --gpus 4 --shm-size 1g -p 8080:80 -v $PWD/data:/data ghcr.io/huggingface/text-generation-inference:latest --model-id OpenAssistant/pythia-12b-sft-v8-7k-steps --num-shard 4 --quantize bitsandbytes > nohup-1.out &

#nohup docker run --gpus 1 --shm-size 1g -p 8081:80 -v $PWD/data:/data ghcr.io/huggingface/text-generation-inference:latest --model-id humarin/chatgpt_paraphraser_on_T5_base --num-shard 1 > nohup-2.out &

#nohup docker run --gpus 1 --shm-size 1g -p 8082:80 -v $PWD/data:/data ghcr.io/huggingface/text-generation-inference:latest --model-id microsoft/DialoGPT-medium --num-shard 1 > nohup-3.out &

#nohup docker run --gpus 1 --shm-size 1g -p 8082:80 -v $PWD/data:/data ghcr.io/huggingface/text-generation-inference:latest --model-id openlm-research/open_llama_7b_700bt_preview --num-shard 1 --quantize bitsandbytes > nohup-3.out &

#nohup docker run --gpus 2 --shm-size 1g -p 8083:80 -v $PWD/data:/data ghcr.io/huggingface/text-generation-inference:latest --model-id TheBloke/OpenAssistant-SFT-7-Llama-30B-HF --num-shard 2 --quantize bitsandbytes > nohup-4.out &

#nohup docker run --gpus 1 --shm-size 1g -p 8082:80 -v $PWD/data:/data ghcr.io/huggingface/text-generation-inference:latest --model-id togethercomputer/RedPajama-INCITE-Chat-7B-v0.1 --num-shard 1 --quantize bitsandbytes > nohup-3.out &
