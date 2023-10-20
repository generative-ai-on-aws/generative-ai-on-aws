# Overview
This repo is based on the following repos:

* https://github.com/huggingface/chat-ui/
* https://github.com/huggingface/text-generation-inference

# Install scripts
This demo uses a single Amazon EC2 g5.12xlarge instance type configured with the latest Nvidia GPU AMI.
```
sudo apt update

# Node
sudo apt-get purge -y nodejs npm
sudo apt-get autoremove

sudo curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.37.2/install.sh | sudo bash
source ~/.bashrc
sudo nvm install node@20
sudo nvm install vite

sudo apt-get install -y libcap2-bin
sudo setcap cap_net_bind_service=+ep /home/ubuntu/.nvm/versions/node/v20.2.0/bin/node

#sudo apt upgrade -y nodejs
#sudo apt upgrade -y npm
#sudo npm install -y npm@latest -g

# Uvicorn
sudo apt install -y uvicorn
sudo pip install -r ~/chat/openai_text_generation_inference_server/requirements.txt

# Mongo
sudo apt-get install -y software-properties-common gnupg apt-transport-https ca-certificates
wget -qO - https://www.mongodb.org/static/pgp/server-5.0.asc | sudo apt-key add -
echo "deb [ arch=amd64,arm64 ] https://repo.mongodb.org/apt/ubuntu focal/mongodb-org/5.0 multiverse" | sudo tee /etc/apt/sources.list.d/mongodb-org-5.0.list
sudo apt-get update
sudo apt-get install -y mongodb-org
mongod --version
sudo systemctl start mongod
sudo systemctl daemon-reload
sudo systemctl status mongod
```

# Setup
Create a `.env.local` file using `.env.template` file.  Below is a portion of the demo's `.env.local` file

```
MODELS=`[
  {
      "name": "Llama2-70b-chat",
      "endpoints": [
        {"url": "http://127.0.0.1:8080", "weight": 100}
      ],
      "userMessageToken": "",
      "userMessageEndToken": " [/INST] ",
      "assistantMessageToken": "",
      "assistantMessageEndToken": " </s><s>[INST] ",
      "preprompt": "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.",
      "chatPromptTemplate" : "<s>[INST] <<SYS>>\n{{preprompt}}\n<</SYS>>\n\n{{#each messages}}{{#ifUser}}{{content}} [/INST] {{/ifUser}}{{#ifAssistant}}{{content}} </s><s>[INST] {{/ifAssistant}}{{/each}}",
      "parameters": {
        "temperature": 0.1,
        "top_p": 0.95,
        "repetition_penalty": 1.2,
        "top_k": 50,
        "truncate": 1000,
        "max_new_tokens": 1024
      }
  },
  {
    "name": "Bedrock_Claude",
    "endpoints": [
      {"url": "http://127.0.0.1:8070/generate_stream"}
    ],
    "userMessageToken": "",
    "userMessageEndToken": "",
    "assistantMessageToken": "",
    "assistantMessageEndToken": "",
    "preprompt": "You are a helpful assistant.",
    "chatPromptTemplate": "\n\n{{preprompt}}{{#each messages}}{{#ifUser}}\n\nHuman:{{content}}\nAssistant:{{/ifUser}}{{#ifAssistant}}{{content}}\n\n</s>{{/ifAssistant}}{{/each}}",
    "parameters": {
      "temperature": 0.9,
      "max_new_tokens": 50,
      "truncate": 1000
    }
  },
  {
    "name": "ChatGPT",
    "endpoints": [
      {"url": "http://127.0.0.1:8060/generate_stream"}
    ],
    "userMessageToken": "User: ",
    "assistantMessageToken": "Assistant: ",
    "messageEndToken": "\n",
    "preprompt": "You are a helpful assistant.",
    "parameters": {
    "temperature": 0.9,
    "max_new_tokens": 50,
    "truncate": 1000
    }
  }
]`
```

# Start
Run `./start-local-server.sh` to start the model server as a Docker container.

Run `./start-chat-ui.sh` to start the NodeJS app.

# Stop
Run `./stop-chat-ui.sh` to stop the NodeJS app.

Run `docker ps` and stop the model server Docker container using the container id.
