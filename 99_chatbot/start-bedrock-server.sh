nohup sudo \
    uvicorn bedrock_text_generation_inference_server.server:app --port 8070 --reload > nohup-bedrock.out &
