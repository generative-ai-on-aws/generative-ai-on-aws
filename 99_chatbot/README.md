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
Edit the [](./env.template) file

# Start
Run [](./start-text-generative-inference.sh) to start the model server as a Docker container

Run [](./start-chat-ui.sh) to start the NodeJS app

# Stop

Run [](./stop-chat-ui.sh) to stop the NodeJS app

Run `docker ps` and stop the docker container using the container id.
