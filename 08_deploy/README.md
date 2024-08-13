# 8장 모델 배포 최적화
[![](../img/gaia_book_cover_sm.png)](https://www.amazon.com/Generative-AI-AWS-Multimodal-Applications/dp/1098159225/)

# 질문&답변
_Q: 가지치기는 어떻게 모델 효율성을 향상시키나요?_

A: 가지치기는 중요도가 낮은 뉴런을 제거하여 모델의 크기를 줄입니다. 이를 통해 모델 성능에 큰 영향을 미치지 않으면서도 추론 시간을 단축하고 메모리 사용량을 줄일 수 있습니다.

_Q: GPTQ를 이용한 사후 학습 양자화란 무엇인가요?_

A: GPTQ(Generalized Poisson Training Quantization)를 이용한 사후 학습 양자화는 학습 후 모델 매개변수의 정밀도를 낮추는 과정입니다. 이를 통해 정확도의 큰 손실 없이 모델 크기를 줄이고 실행 속도를 높일 수 있습니다.

_Q: A/B 테스트와 섀도 배포는 배포 전략에서 어떤 차이가 있나요?_

A: A/B 테스트는 새 모델의 성능을 기존 모델과 비교하기 위해 트래픽의 일부를 새 모델로 전송합니다. 반면 섀도 배포는 실제 사용자 트래픽을 전송하지 않고 새 모델을 기존 모델과 병렬로 실행하여 주로 테스트와 평가 목적으로 사용됩니다.

_Q: 모델 배포 최적화가 전반적인 성능과 확장성에 어떤 영향을 미치나요?_

A: 모델 압축, 효율적인 하드웨어 활용, 부하 분산 등의 모델 배포 최적화를 통해 성능을 크게 개선하고, 비용을 절감하며, 다양한 부하를 처리할 수 있는 확장성을 확보할 수 있습니다.

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
