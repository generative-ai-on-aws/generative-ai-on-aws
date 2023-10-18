nohup sudo \
    OPENAI_API_KEY=<OPENAI_API_KEY> \
    uvicorn server:app --port 8060 --reload > nohup-openai.out &
