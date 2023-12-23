# Chapter 6: Parameter-Efficient Fine-Tuning (PEFT)
[![](../img/gaia_book_cover_sm.png)](https://www.amazon.com/Generative-AI-AWS-Multimodal-Applications/dp/1098159225/)

# Questions and Answers

_Q: In what scenarios is PEFT preferable over traditional fine-tuning methods?_

A: PEFT is preferable in scenarios where model efficiency is crucial, and only specific parts of the model need adaptation, reducing the computational resources required. 

_Q: How does PEFT impact the adaptability of Generative AI models?_

A: PEFT enhances the adaptability of Generative AI models by allowing efficient fine-tuning of specific parts, reducing the need to train the entire model. 

_Q: What is the significance of target modules and layers in PEFT?_

A: Target modules and layers in PEFT are specific parts of the model that are fine-tuned, allowing for efficient training and adaptation without modifying the entire model. 

_Q: What are LoRA and QLoRA PEFT techniques, and how do they function?_

A: LoRA (Low-Rank Adaptation) is a technique applied to linear layers of a model to adapt it with minimal changes. QLoRA (Quantized LoRA) involves additional quantization for more efficiency.

_Q: How does the rank of LoRA influence model performance?_

A: The rank of LoRA, which refers to the number of parameters added, influences the balance between model adaptability and efficiency. Higher ranks can lead to better performance but at the cost of efficiency.

_Q: How does maintaining separate LoRA adapters benefit the model?_

A: Maintaining separate LoRA adapters allows for the original model to remain unchanged. These adapters can be merged with the original model or kept separate for flexibility.

_Q: What is prompt tuning, and how does it differ from soft prompts?_

A: Prompt tuning involves adjusting the input prompts to guide the model's output. Soft prompts refer to virtual tokens generated to achieve similar effects. The document does not elaborate on their differences.

_Q: How do performance comparisons between full fine-tuning and PEFT/LoRA help model optimization?_

A: Performance comparisons between full fine-tuning and LoRA help in understanding the trade-offs between model efficiency and adaptability, guiding optimization decisions.

# Chapters
* [Chapter 1](/01_intro) - Generative AI Use Cases, Fundamentals, Project Lifecycle
* [Chapter 2](/02_prompt) - Prompt Engineering and In-Context Learning
* [Chapter 3](/03_foundation) - Large-Language Foundation Models
* [Chapter 4](/04_optimize) - Quantization and Distributed Computing
* [Chapter 5](/05_finetune) - Fine-Tuning and Evaluation
* [Chapter 6](/06_peft) - Parameter-efficient Fine Tuning (PEFT)
* [Chapter 7](/07_rlhf) - Fine-tuning using Reinforcement Learning with RLHF
* [Chapter 8](/08_deploy) - Optimize and Deploy Generative AI Applications
* [Chapter 9](/09_rag) - Retrieval Augmented Generation (RAG) and Agents
* [Chapter 10](/10_multimodal) - Multimodal Foundation Models
* [Chapter 11](/11_diffusers) - Controlled Generation and Fine-Tuning with Stable Diffusion
* [Chapter 12](/12_bedrock) - Amazon Bedrock Managed Service for Generative AI

# Related Resources
* YouTube Channel: https://youtube.generativeaionaws.com
* Generative AI on AWS Meetup (Global, Virtual): https://meetup.generativeaionaws.com
* Generative AI on AWS O'Reilly Book: https://www.amazon.com/Generative-AI-AWS-Multimodal-Applications/dp/1098159225/
* Data Science on AWS O'Reilly Book: https://www.amazon.com/Data-Science-AWS-End-End/dp/1492079391/
