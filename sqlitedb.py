import sqlite3

# 데이터베이스 파일 경로
db_path = "result/chroma.sqlite3"

# SQLite 데이터베이스 연결
connection = sqlite3.connect(db_path)

# 커서 생성
cursor = connection.cursor()

# SQL 쿼리 실행 예시
cursor.execute("SELECT * FROM sqlite_master WHERE type='table';")
data = cursor.fetchall()

for d in data:
    print(data[0])
# 작업 완료 후 연결 닫기
cursor.close()
connection.close()