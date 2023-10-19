nohup sudo \
    uvicorn openai_text_generation_inference_server.server:app --port 8060 --reload > nohup-openai.out &
