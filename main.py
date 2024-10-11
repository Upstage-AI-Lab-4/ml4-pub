# main.py

from api_module import create_app
from database import init_db
import uvicorn

def main():
    # 데이터베이스 초기화
    init_db()

    # FastAPI 애플리케이션 생성
    app = create_app()

    # 애플리케이션 실행
    uvicorn.run(app)

if __name__ == "__main__":
    main()
