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
import tempfile
import torchaudio
from botocore.exceptions import NoCredentialsError


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def model_fn(model_dir):
    """
    Load and return the model
    """
    model = whisper.load_model(os.path.join(model_dir, 'base.pt'))
    model = model.to(DEVICE)
    print(f'whisper model has been loaded to this device: {model.device.type}')
    return model

def transform_fn(model, request_body, request_content_type, response_content_type="application/json"):
    """
    Transform the input data and generate a transcription result
    """
    logging.info("Check out the request_body type: %s", type(request_body))
    start_time = time.time()
    
    file = io.BytesIO(request_body)
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(file.read())

    logging.info("Start generating the transcription ...")
    result = model.transcribe(tfile.name)
    logging.info("Transcription generation completed.")

    end_time = time.time()
    elapsed_time = end_time - start_time
    logging.info("Elapsed time: %s seconds", elapsed_time)

    return json.dumps(result), response_content_type