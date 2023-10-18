from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from fastapi.responses import StreamingResponse
import json
import random
#import openai
#import bedrock
import asyncio
import os
import boto3

bedrock = boto3.client('bedrock', 'us-east-1')
print(bedrock.list_foundation_models())

app = FastAPI()

# AWS Credentials must be available through normal ways 

DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 100

class RequestBody(BaseModel):
    inputs: str
    parameters: Optional[dict]

async def get_bedrock_stream_data(request):
    prompt_data = request.inputs

    # Anthropic: "prompt"
    # Titan: "inputText"
    body = json.dumps({"inputText": prompt_data})
    modelId = 'amazon.titan-tg1-large' # change this to use a different version from the model provider
    #modelId = 'anthropic.claude-v1'
    accept = 'application/json'
    contentType = 'application/json'

    response = bedrock.invoke_model(body=body, modelId=modelId, accept=accept, contentType=contentType)

    end = "\n\n" # </s>"
    response_body = json.loads(response.get('body').read())
    output_text = response_body.get('results')[0].get('outputText')
    final_text = None 
    details = None
    special = False
    tok = {
        "token": {
            "id": random.randrange(0, 2**32),
            "text": output_text,
            "logprob": 0,
            "special": special,
        },
        "generated_text": final_text,
        "details": details
    }
    json_string = json.dumps(tok,separators=(',', ':'))

    response_stream = f"data:{json_string}"
    yield response_stream + end
    await asyncio.sleep(0)  # Allow other tasks to run 


@app.post("/generate_stream")
async def chat_completions(request: RequestBody):
    return StreamingResponse(get_bedrock_stream_data(request), media_type="text/event-stream", headers={"Content-Type": "text/event-stream"})
