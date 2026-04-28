# 데이터 수집 배치 스크립트

## 역할
DB에서 학습 데이터 추출 → CSV 생성 → S3 업로드 → SSE 로그 발행

## 전체 흐름
```
1. DB(emails + email_analysis_results) 조인하여 데이터 추출
2. CSV 파일 생성 (emailId, from, subject, body, domain, intent)
3. S3 업로드 (s3://capstone-gachon/dataset/dataset_new.csv)
4. 진행 로그 → x.sse.fanout 발행
5. 완료/실패 이벤트 → q.2app.training 발행
```

## 필수 환경변수

| 변수명 | 설명 | 기본값 |
|--------|------|--------|
| DB_HOST | DB 호스트 | 192.168.3.10 |
| DB_PORT | DB 포트 | 3306 |
| DB_USER | DB 사용자 | capstone |
| DB_PASSWORD | DB 비밀번호 | capstone |
| DB_NAME | DB 이름 | email_agent |
| AWS_ACCESS_KEY_ID | AWS Access Key | 필수 |
| AWS_SECRET_ACCESS_KEY | AWS Secret Key | 필수 |
| AWS_REGION | AWS 리전 | ap-northeast-2 |
| S3_BUCKET | S3 버킷명 | capstone-gachon |
| S3_DATASET_KEY | S3 업로드 경로 | dataset/dataset_new.csv |
| RABBITMQ_HOST | RabbitMQ 호스트 | 192.168.2.20 |
| RABBITMQ_PORT | RabbitMQ 포트 | 30672 |
| RABBITMQ_USERNAME | RabbitMQ 사용자 | admin |
| RABBITMQ_PASSWORD | RabbitMQ 비밀번호 | admin1234! |
| JOB_ID | 백엔드에서 생성한 job_id | 필수 |
| ADMIN_USER_ID | 관리자 user_id (SSE용) | 0 |

## 로컬 테스트

```bash
pip install -r requirements.txt

export JOB_ID=test-job-001
export ADMIN_USER_ID=54
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret

python dataset_batch.py
```

## k8s Job 실행 시
백엔드가 k8s Job manifest로 실행.
환경변수는 k8s Secret/ConfigMap으로 주입.

## CSV 컬럼 구조
```
emailId,from,subject,body,domain,intent
train_1,sender@example.com,제목,본문내용,Finance,세금계산서 요청
```
