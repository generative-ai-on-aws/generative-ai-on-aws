"""
server.py
Author: Gene Ruebsamen

This file is part of OpenAI-Text-Generation-Inference-Server.

OpenAI-Text-Generation-Inference-Server is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

OpenAI-Text-Generation-Inference-Server is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with OpenAI-Text-Generation-Inference-Server. If not, see <https://www.gnu.org/licenses/>.
"""

from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from fastapi.responses import StreamingResponse
from dotenv import load_dotenv
import json
import random
import openai
import asyncio
import os

load_dotenv()

app = FastAPI()

openai.api_key = os.environ['OPENAI_API_KEY']
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 100

class RequestBody(BaseModel):
    inputs: str
    parameters: Optional[dict]

async def get_openai_stream_data(request):
    events = await openai.ChatCompletion.acreate(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": request.inputs}],
        stream=True,
        temperature = request.parameters['temperature'] if 'parameters' in request else DEFAULT_TEMPERATURE,
        max_tokens = request.parameters['max_tokens'] if 'parameters' in request else DEFAULT_MAX_TOKENS,
        )
    
    gen_text = ""
    end = "\n\n"
    tok_cnt = 0
    async for event in events:
        #print(event)
        try:      
            content = event['choices'][0]['delta']['content']
        except KeyError:
            content = None
        finish_reason = event['choices'][0]['finish_reason']
        tok_cnt += 1
        if content or finish_reason != None:
            if content:
                gen_text += content
            final_text = None
            details = None
            special = False
            if finish_reason!=None:
                final_text = gen_text
                special = True
                details = {"finish_reason":finish_reason,"generated_tokens":tok_cnt-1,"seed":None}
                end = "\n\n\n"
            tok = {
                "token": {
                    "id": random.randrange(0, 2**32),
                    "text": content,
                    "logprob": 0,
                    "special": special,
                },
                "generated_text": final_text,
                "details": details
            }
            json_string = json.dumps(tok,separators=(',', ':'))
            result = f"data:{json_string}"
            print(result)
            yield result + end
            await asyncio.sleep(0)  # Allow other tasks to run

@app.post("/generate_stream")
async def chat_completions(request: RequestBody):
    return StreamingResponse(get_openai_stream_data(request), media_type="text/event-stream", headers={"Content-Type": "text/event-stream"})
