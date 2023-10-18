from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from fastapi.responses import StreamingResponse
import json
import random
import asyncio
import os
import boto3

bedrock = boto3.client('bedrock', 'us-east-1')
print(bedrock.list_foundation_models())

prompt_data ="""Write me a blog about making strong business decisions as a leader"""#If you'd like to try your own prompt, edit this parameter!

body = json.dumps({"inputText": prompt_data})
modelId = 'amazon.titan-tg1-large' # change this to use a different version from the model provider
accept = 'application/json'
contentType = 'application/json'

response = bedrock.invoke_model(body=body, modelId=modelId, accept=accept, contentType=contentType)
response_body = json.loads(response.get('body').read())

print(response_body.get('results')[0].get('outputText'))
