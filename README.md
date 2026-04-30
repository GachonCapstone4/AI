# 업무 이메일 자동화 AI 시스템

> 캡스톤 디자인 프로젝트 — AI 기반 업무 이메일 자동 분류 및 응답 자동화 서비스

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

## AI 모델 구조

| 단계 | 모델 | 역할 |
|------|------|------|
| 임베딩 | SBERT (paraphrase-multilingual-MiniLM-L12-v2) | 이메일 텍스트 → 벡터 변환 |
| 1차 분류 | Logistic Regression | 7개 Domain 분류 |
| 2차 분류 | Logistic Regression (Domain별) | 30개 Intent 분류 |
| 요약/추출 | GPT-4o-mini | 이메일 요약 / 일정 추출 |
| 답장 생성 | Claude 3.5 Sonnet | 답장 템플릿 초안 생성 |

---

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

## Kubernetes Job 실행 가이드

이 가이드는 172.16.2.10 Linux EC2에서 dataset_batch.py 기반 Kubernetes Job을 실행하는 기준입니다. Kubernetes Job manifest는 --manifest-path로 전달합니다.

dataset_batch.py는 이미지의 CMD를 통해 기본 실행됩니다. manifest에는 command와 args를 지정하지 않습니다.

실행 파라미터는 컨테이너 환경변수로 주입합니다.

    JOB_ID
    ADMIN_USER_ID

AWS Credential은 Kubernetes Secret job-secret에서 envFrom으로 주입합니다.

    AWS_ACCESS_KEY_ID
    AWS_SECRET_ACCESS_KEY

성공과 실패는 Kubernetes Job exit code 기준입니다. 성공 시 exit 0과 q.2app.training COMPLETED 이벤트를 발행하고, 실패 시 exit 1과 q.2app.training FAILED 이벤트를 발행합니다. 로그는 stdout 기준이며 kubectl logs로 확인합니다. 관리자 화면 실시간 표시는 x.sse.fanout 연동 예정입니다.

예시 manifest:

    cat manifests/dataset-batch.yaml

kubectl 권한 확인:

    kubectl auth can-i create jobs -n admin
    kubectl auth can-i get pods -n admin
    kubectl auth can-i get pods/log -n admin
    kubectl auth can-i delete jobs -n admin

dry-run:

    cd ~/Capstone_AI2
    python -m launcher.run --job-id dataset-batch-001 --job-type k8s_job --dry-run --manifest-path manifests/dataset-batch.yaml

실제 Job 생성:

    cd ~/Capstone_AI2
    python -m launcher.run --job-id dataset-batch-001 --job-type k8s_job --manifest-path manifests/dataset-batch.yaml

상태 확인:

    kubectl get jobs -n admin
    kubectl get pods -n admin -l job-name=dataset-batch

로그 확인:

    kubectl logs -n admin job/dataset-batch

삭제:

    kubectl delete job -n admin dataset-batch

manifest에 metadata.namespace가 없으면 런처가 기본값 admin을 manifest 내부에 채워 넣습니다. manifest에 namespace가 이미 있으면 덮어쓰지 않습니다.

---

## 라이선스

본 프로젝트는 캡스톤 디자인 학술 목적으로 제작되었습니다.
