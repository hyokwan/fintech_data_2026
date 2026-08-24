# %% [markdown]
# # 라이브러리 불러오기

# %%
# FastAPI 앱 실행 위한 비동기 서버 라이브러리
import uvicorn

# REST API 서버 생성 위한 FastAPI 핵심 라이브러리
from fastapi import FastAPI

# 머신러닝 모델 파일 저장/불러오기 위한 라이브러리
import joblib

# 데이터 처리 및 분석을 위한 라이브러리
import pandas as pd
import numpy as np

# 인터페이스 데이터 관리를 위한 라이브러리
from pydantic import BaseModel

# 다른 서버에서 오는 요청을 허용하기 위한 CORS 미들웨어
from fastapi.middleware.cors import CORSMiddleware


# %%
import joblib
import pandas as pd

# %% [markdown]
# ### 1. CORS 설정

# %%
# FastAPI 애플리케이션 생성 (API 문서 제목 설정)
app = FastAPI(title="ML API")

# 모든 출처(origin)에서의 접근을 허용하기 위한 설정
origins = ["*"]

# CORS(Cross-Origin Resource Sharing) 미들웨어 추가
# 모든 도메인, 쿠키,인증정보, HTTP 메소드, HTTP 헤더 허용
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],  
    allow_headers=origins,
)


# %% [markdown]
# ### 2. 모델 불러오기

# %%
# 모델 불러오기
modelDump = joblib.load("mlcore.dump")
loadedModel = modelDump["model"]
features = modelDump["features"]
label = modelDump["label"]
preprocessing = modelDump["preprocessing"]
print(f" loadedModel = {loadedModel} \n features = {features}" )
# 테스트데이터 설정
inHclus = 4
inPropercent = 0.0
inPromotion = "N"
inHoliday = "N" 
# 입력 데이터 전처리
ynLabel = preprocessing[2]
ynLabel.classes_
inPreProPromotion = int( ynLabel.transform( [inPromotion] )[0] )
ynLabel2 = preprocessing[3]
inPreProHoliday = int( ynLabel2.transform( [inHoliday] )[0] )
# 전처리 완료된 데이터셋 테스트 데이터프레임으로 변환
inDf = pd.DataFrame( [[ inHclus, inPropercent,inPreProPromotion, inPreProHoliday ]])
# 예측 테스트
loadedModel.predict(inDf)

# %% [markdown]
# ### 3. 인터페이스 데이터 정의

# %%
# API 요청으로 들어오는 입력 데이터를 정의하는 Pydantic 모델
# 머신러닝 예측에 사용할 입력 변수들의 자료형을 명시
class InDataset(BaseModel):
    inHclus : int
    inPropercent : float
    inPromotion : str
    inHoliday : str


# %% [markdown]
# ### 4. 엔드포인트 정의

# %%
x = InDataset
x.inHclus = 0
x.inPropercent = 0.3 
x.inPromotion = "Y"
x.inHoliday = "Y"
# 테스트데이터 설정
inHclus = x.inHclus
inPropercent = x.inPropercent
inPromotion = x.inPromotion
inHoliday = x.inHoliday
# 입력 데이터 전처리
ynLabel = preprocessing[2]
ynLabel.classes_
inPreProPromotion = int( ynLabel.transform( [inPromotion] )[0] )
ynLabel2 = preprocessing[3]
inPreProHoliday = int( ynLabel2.transform( [inHoliday] )[0] )
# 전처리 완료된 데이터셋 테스트 데이터프레임으로 변환
inDf = pd.DataFrame( [[ inHclus, inPropercent,inPreProPromotion, inPreProHoliday ]])
# 예측 테스트
predictValue = int( loadedModel.predict(inDf)[0] )
predictValue
print(predictValue)

# %%
# POST 방식으로 예측 요청을 받는 API 엔드포인트
@app.post("/predictBase", status_code=200)
async def predict_tf(x: InDataset):
    # x = InDataset
    # x.inHclus = 0
    # x.inPropercent = 0.3 
    # x.inPromotion = "Y"
    # x.inHoliday = "Y"
    # 테스트데이터 설정
    inHclus = x.inHclus
    inPropercent = x.inPropercent
    inPromotion = x.inPromotion
    inHoliday = x.inHoliday
    # 입력 데이터 전처리
    ynLabel = preprocessing[2]
    ynLabel.classes_
    inPreProPromotion = int( ynLabel.transform( [inPromotion] )[0] )
    ynLabel2 = preprocessing[3]
    inPreProHoliday = int( ynLabel2.transform( [inHoliday] )[0] )
    # 전처리 완료된 데이터셋 테스트 데이터프레임으로 변환
    inDf = pd.DataFrame( [[ inHclus, inPropercent,inPreProPromotion, inPreProHoliday ]])
    # 예측 테스트
    predictValue = int( loadedModel.predict(inDf)[0] )
    predictValue
    print(predictValue)
    return {"prediction":predictValue}


# %%
# GET 방식으로 서버 상태 확인 엔드포인트
@app.get("/")
async def root():
    return {"message":"onine"}


# %% [markdown]
# ### 5. 서버 오픈

# %%
if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",  # 로컬 또는 내부 네트워크 접근 허용
        port=9999,
        reload=True     # 개발 중 자동 재시작
    )



