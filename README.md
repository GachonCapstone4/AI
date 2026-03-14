# 업무 이메일 자동화 AI 시스템

> 캡스톤 디자인 프로젝트 — AI 기반 업무 이메일 자동 분류 및 응답 자동화 서비스

- 저는 현재 AI를 맡고 있으며, MLOps까지 진행할 예정입니다.
---

## 프로젝트 개요

Gmail API로 수신된 업무 이메일을 AI가 자동으로 분류하고,
LLM 기반 답장 초안 생성 및 일정 등록까지 자동화하는 AI Agent 서비스입니다.

---

## 전체 시스템 흐름
```
Gmail API
→ 백엔드 1차 필터링
→ AI 서버 (이 레포)
  → 전처리 (subject + body → email_text)
  → SBERT 임베딩 생성
  → 1차 분류: Domain (Logistic Regression)
  → 2차 분류: Intent (Logistic Regression)
  → LLM 후처리
    - GPT   : 이메일 요약 / 일정 추출
    - Claude : 답장 템플릿 초안 생성
  → Google Calendar 등록 후보 생성
```

---

## 폴더 구조
```
Capstone_ai/
├── notebooks/
│   └── project.ipynb       # 전체 파이프라인 실행 노트북
├── src/
│   ├── config.py           # 경로 및 하이퍼파라미터 설정
│   ├── data_utils.py       # 데이터 로드 / Pair 생성
│   ├── train_sbert.py      # SBERT Fine-tuning / 임베딩 생성
│   ├── train_domain.py     # Domain 분류기 학습
│   ├── train_intent.py     # Intent 분류기 학습
│   ├── evaluation.py       # F1 / Confusion Matrix 평가
│   └── inference.py        # 추론 파이프라인
├── data/                   # 데이터 저장 (gitignore)
├── models/                 # 학습된 모델 저장 (gitignore)
├── outputs/
│   ├── figures/            # Confusion Matrix 이미지
│   ├── reports/            # 평가 리포트
│   └── logs/               # 학습 로그
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 🤖 AI 모델 구조

| 단계 | 모델 | 역할 |
|------|------|------|
| 임베딩 | SBERT (paraphrase-multilingual-MiniLM-L12-v2) | 이메일 텍스트 → 벡터 변환 |
| 1차 분류 | Logistic Regression | 7개 Domain 분류 |
| 2차 분류 | Logistic Regression (Domain별) | 31개 Intent 분류 |
| 요약/추출 | GPT-4o-mini | 이메일 요약 / 일정 추출 |
| 답장 생성 | Claude 3.5 Sonnet | 답장 템플릿 초안 생성 |

### Domain / Intent 구조

| Domain | Intent 예시 |
|--------|------------|
| Sales | 견적 요청, 계약 문의, 가격 협상, 제안 요청, 미팅 일정 조율 |
| Marketing & PR | 협찬 제안, 광고 문의, 보도자료 요청, 인터뷰 요청 |
| HR | 채용 문의, 면접 일정 조율, 휴가 신청, 증명서 발급 |
| Finance | 세금계산서 요청, 비용 처리 문의, 입금 확인, 정산 문의 |
| Customer Support | 불만 접수, 기술 지원 요청, 환불 요청, 사용법 문의 |
| IT/Ops | 시스템 오류 보고, 계정 생성 요청, 접근 권한 변경 |
| Admin | 공지 전달, 내부 보고, 자료 요청, 협조 요청 |



---

## 기술 스택

![Python](https://img.shields.io/badge/Python-3.x-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.3-orange)
![SentenceTransformers](https://img.shields.io/badge/SentenceTransformers-2.7-green)
![Scikit--learn](https://img.shields.io/badge/ScikitLearn-1.4-yellow)

---

## 라이선스

본 프로젝트는 캡스톤 디자인 학술 목적으로 제작되었습니다.
