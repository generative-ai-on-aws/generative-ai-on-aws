# 6장 효율적인 매개변수 미세 조정(PEFT)
[![](../img/gaia_book_cover_sm.png)](https://www.amazon.com/Generative-AI-AWS-Multimodal-Applications/dp/1098159225/)

# 질문&답변

_Q: 어떤 시나리오에서 PEFT가 전통적인 미세 조정 방법보다 선호되나요?_

A: PEFT는 모델 효율성이 중요하고 모델의 특정 부분만 적응이 필요한 시나리오에서 선호됩니다. 이는 필요한 컴퓨팅 자원을 줄여줍니다. 

_Q: PEFT는 생성형 AI 모델의 적응성에 어떤 영향을 미치나요?_

A: PEFT는 전체 모델을 학습할 필요 없이 특정 부분만 효율적으로 미세 조정할 수 있게 함으로써 생성형 AI 모델의 적응성을 향상시킵니다. 

_Q: PEFT에서 목표 모듈과 레이어의 중요성은 무엇인가요?_

A: PEFT의 목표 모듈과 레이어는 미세 조정되는 모델의 특정 부분으로, 전체 모델을 수정하지 않고도 효율적인 학습과 적응을 가능하게 합니다. 

_Q: LoRA와 QLoRA PEFT 기법은 무엇이며 어떻게 작동하나요?_

A: LoRA(저순위 적응)는 모델의 선형 레이어에 적용되어 최소한의 변경으로 모델을 적응시키는 기법입니다. QLoRA(양자화된 LoRA)는 더 높은 효율성을 위해 추가적인 양자화를 포함합니다.

_Q: LoRA의 순위는 모델 성능에 어떤 영향을 미치나요?_

A: LoRA의 순위는 추가되는 매개변수의 수를 나타내며, 모델의 적응성과 효율성 사이의 균형에 영향을 줍니다. 높은 순위는 더 나은 성능을 가져올 수 있지만 효율성이 떨어질 수 있습니다.

_Q: LoRA 어댑터를 별도로 유지하는 것이 모델에 어떤 이점을 주나요?_

A: LoRA 어댑터를 별도로 유지하면 원본 모델을 변경하지 않고 유지할 수 있습니다. 이러한 어댑터는 원본 모델과 병합하거나 유연성을 위해 별도로 유지할 수 있습니다.

_Q: 프롬프트 튜닝이란 무엇이며 소프트 프롬프트와 어떻게 다른가요?_

A: 프롬프트 튜닝은 모델의 출력을 유도하기 위해 입력 프롬프트를 조정하는 것입니다. 소프트 프롬프트는 비슷한 효과를 얻기 위해 생성된 가상 토큰을 말합니다. 문서에는 이들의 차이점에 대한 자세한 설명이 없습니다.

_Q: 완전 미세 조정과 PEFT/LoRA 간의 성능 비교가 모델 최적화에 어떤 도움을 주나요?_

A: 완전 미세 조정과 LoRA 간의 성능 비교는 모델 효율성과 적응성 사이의 트레이드오프를 이해하는 데 도움을 주어 최적화 결정을 안내합니다.

# 목차
* [1장](/01_intro) - 생성형 AI 활용 사례, 기본 사항 및 프로젝트 생명 주기
* [2장](/02_prompt) - 프롬프트 엔지니어링과 콘텍스트 내 학습
* [3장](/03_foundation) - 대형 언어 파운데이션 모델
* [4장](/04_optimize) - 메모리와 연산 최적화
* [5장](/05_finetune) - 미세 조정 및 평가
* [6장](/06_peft) - 효율적인 매개변수 미세 조정(PEFT)
* [7장](/07_rlhf) - 인간 피드백을 통한 강화 학습으로 미세 조정(RLHF)
* [8장](/08_deploy) - 모델 배포 최적화
* [9장](/09_rag) - RAG와 에이전트를 활용한 맥락 인식 추론 애플리케이션
* [10장](/10_multimodal) - 멀티모달 파운데이션 모델
* [11장](/11_diffusers) - 스테이블 디퓨전을 통한 생성 제어와 미세 조정
* [12장](/12_bedrock) - 아마존 베드록: 생성형 AI 관리형 서비스

# Related Resources
* YouTube Channel: https://youtube.generativeaionaws.com
* Generative AI on AWS Meetup (Global, Virtual): https://meetup.generativeaionaws.com
* Generative AI on AWS O'Reilly Book: https://www.amazon.com/Generative-AI-AWS-Multimodal-Applications/dp/1098159225/
* Data Science on AWS O'Reilly Book: https://www.amazon.com/Data-Science-AWS-End-End/dp/1492079391/
