# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0


import os
import io
import sys
import time
import json
import logging

import whisper
import torch
import boto3
import ffmpeg
import torchaudio
import tempfile
from transformers import pipeline

device = "cuda:0" if torch.cuda.is_available() else "cpu"
chunk_length_s = int(os.environ.get('chunk_length_s'))

def model_fn(model_dir):
    model = pipeline(
        "automatic-speech-recognition",
        model=model_dir,
        chunk_length_s=chunk_length_s,
        device=device,
        )
    return model


def transform_fn(model, request_body, request_content_type, response_content_type="application/json"):
     
    logging.info("Check out the request_body type: %s", type(request_body))
    
    start_time = time.time()
    
    file = io.BytesIO(request_body)
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(file.read())

    logging.info("Start to generate the transcription ...")
    result = model(tfile.name, batch_size=8)["text"]
    
    logging.info("Upload transcription results back to S3 bucket ...")
    
    # Calculate the elapsed time
    end_time = time.time()
    elapsed_time = end_time - start_time
    logging.info("The time for running this program is %s", elapsed_time)
    
    return json.dumps(result), response_content_type   