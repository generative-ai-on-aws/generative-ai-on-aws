# Chapter 2: Prompt Engineering and In-Context Learning
[![](../img/gaia_book_cover_sm.png)](https://www.amazon.com/Generative-AI-AWS-Multimodal-Applications/dp/1098159225/)

# Questions and Answers

_Q: How do prompts and completions work in Generative AI?_

A: Prompts in Generative AI involve text-based input, including instructions, context, and constraints for a task. The AI model responds with a 'completion' that could be text, image, video, or audio, depending on the model's training.

_Q: What role do tokens play in prompt engineering?_

A: In prompt engineering, tokens are crucial as generative models convert text-based prompts and completions into sequences of tokens or word fragments. These tokens represent words and enable the model to process and generate language-based responses.

_Q: Can you explain the concept of prompt structure?_

A: Prompt structure typically includes instruction, context, input data, and output indicator. Effective prompt engineering involves creating a prompt structure that guides the model towards the desired output, with clarity, specificity, and context.

_Q: What is in-context learning in Generative AI?_

A: In-context learning in Generative AI involves passing multiple prompt-completion pairs along with the prompt input. This technique temporarily alters the model's behavior for that request, encouraging it to respond in ways similar to the examples provided in the context.

_Q: How does few-shot inference differ from zero-shot and one-shot inference?_

A: Few-shot inference involves providing multiple examples within the prompt context for the model to learn from. In contrast, one-shot inference uses a single example, and zero-shot inference uses no examples, relying on the model's preexisting knowledge and generalization capabilities.

_Q: What are some best practices for in-context learning?_

A: For effective in-context learning, start with zero-shot inference and progress to one-shot or few-shot if needed. Ensure the mix of examples is consistent and appropriate for the task. The context shouldn't exceed the model's input size or context window.

_Q: What are some effective prompt-engineering techniques?_

A: Effective prompt-engineering techniques include being clear and concise, being creative, moving the instruction to the end of the prompt for longer texts, clearly conveying the subject, using explicit directives, avoiding negative formulations, including context and few-shot examples, specifying response size, and providing a specific response format.

_Q: What inference configuration parameters are critical in prompt engineering?_

A: Critical inference configuration parameters in prompt engineering include temperature and top k, which control the model's creativity in generating content.

_Q: How does prompt structure affect the performance of a Generative AI model?_

A: Prompt structure affects a Generative AI model's performance by guiding it towards the desired output.

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
