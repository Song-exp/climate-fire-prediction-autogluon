import pandas as pd
from autogluon.tabular import TabularDataset, TabularPredictor
import os
import numpy as np
import math

folder_path = 'C:/빅데이터 활용 미래 사회문제 해결 아이디어 해커톤/전처리 후 데이터'  # 폴더 경로 지정
files = os.listdir(folder_path)

file_name = [f for f in files if os.path.isfile(os.path.join(folder_path, f))]


def calculate_wind_components(wind_speed, wind_direction):
    # Convert wind direction from degrees to radians for trigonometric functions
    wind_direction_radians = math.radians(wind_direction)

    # Calculate U component (West-East)
    u_component = wind_speed * math.cos(wind_direction_radians)

    # Calculate V component (South-North)
    v_component = wind_speed * math.sin(wind_direction_radians)

    return u_component, v_component

columns_Z = ['사망', '부상', '재산피해소계']

columns_X = ['기온(°C)', '풍향(deg)', '풍속(m/s)', '강수량(mm)', '현지기압(hPa)', '해면기압(hPa)', '습도(%)']

columns_X1 =['경보종류_강풍경보', '경보종류_강풍주의보','경보종류_건조경보', '경보종류_건조주의보', '경보종류_대설경보',
 '경보종류_대설주의보', '경보종류_태풍경보', '경보종류_태풍주의보', '경보종류_폭염경보', '경보종류_폭염주의보',
 '경보종류_폭풍해일주의보', '경보종류_풍랑경보', '경보종류_풍랑주의보', '경보종류_한파경보', '경보종류_한파주의보',
 '경보종류_호우경보','경보종류_호우주의보','경보종류_황사경보']

columns_Y = ['화재유형_건축,구조물', '화재유형_기타(쓰레기 화재등)', '화재유형_선박,항공기', '화재유형_위험물,가스제조소등', '화재유형_임야', '화재유형_자동차,철도차량']
columns_Y1 = ['발화열원_기타', '발화열원_담뱃불, 라이터불', '발화열원_마찰, 전도, 복사', '발화열원_미상', '발화열원_불꽃, 불티',
              '발화열원_자연적 발화열', '발화열원_작동기기', '발화열원_폭발물, 폭죽', '발화열원_화학적 발화열','발화요인_가스누출(폭발)',
              '발화요인_교통사고', '발화요인_기계적 요인', '발화요인_기타', '발화요인_미상', '발화요인_방화', '발화요인_방화의심',
              '발화요인_부주의', '발화요인_자연적인 요인', '발화요인_전기적 요인', '발화요인_제품결함', '발화요인_화학적 요인']
columns_Y2 = ['최초착화물_가구', '최초착화물_가연성가스', '최초착화물_간판,차양막등', '최초착화물_기타', '최초착화물_미상',
              '최초착화물_식품', '최초착화물_쓰레기류', '최초착화물_위험물등', '최초착화물_자동차,철도차량,선박,항공기',
              '최초착화물_전기,전자', '최초착화물_종이,목재,건초등', '최초착화물_침구,직물류', '최초착화물_합성수지']

columns_Y3 = ['장소_교육시설', '장소_기타', '장소_기타서비스', '장소_문화재시설', '장소_산업시설', '장소_생활서비스',
              '장소_선박,항공기', '장소_운수자동차시설', '장소_위험물,가스제조소', '장소_의료,복지시설', '장소_임야',
              '장소_자동차,철도차량', '장소_주거', '장소_집합시설', '장소_판매,업무시설']

result_dict = {}

for i in range(len(file_name)):
    file_path = os.path.join(folder_path, files[i])
    try:
        df = pd.read_csv(file_path, index_col=0)
    except UnicodeDecodeError:
        df = pd.read_csv(file_path, encoding='cp949', index_col=0)

    df_model = df[columns_X].copy()

    df_model.loc[:, '화재합계'] = df[columns_Y].sum(axis=1)

    fire_count_dict = dict(round(df_model['화재합계'].value_counts()/len(df_model['화재합계']),5))

    # fire_cat_dict = {}

    # for key, value in fire_count_dict.items():
    #     if key == 0 :
    #         fire_cat_dict[key] = 0
    #     elif (key == 1) and (fire_count_dict[key] < 0.1):
    #         fire_cat_dict[key] = 0
    #     elif (key == 2) and (fire_count_dict[key] < 0.1) and (fire_count_dict[key] < fire_count_dict[key+1]):
    #         fire_cat_dict[key] = 0
    #     elif value >= 0.1:   # 10% 이상
    #         fire_cat_dict[key] = 1
    #     elif value >= 0.01: # 10% 미만 1% 이상
    #         fire_cat_dict[key] = 2
    #     else:               # 1% 미만
    #         fire_cat_dict[key] = 3

    # df_model['화재합계'] = df_model['화재합계'].map(fire_cat_dict)


    df_model["U Component"], df_model["V Component"] = zip(*df_model.apply(lambda row: calculate_wind_components(row["풍속(m/s)"], row["풍향(deg)"]), axis=1))
    df_model = df_model.drop(["풍속(m/s)", "풍향(deg)"], axis=1)

    df_model["기압 비율"] = df_model["현지기압(hPa)"] / df_model["해면기압(hPa)"]
    df_model = df_model.drop(["현지기압(hPa)", "해면기압(hPa)"], axis=1)

    df_model.index = pd.to_datetime(df_model.index)
    print(set(df_model['화재합계']))

    df_test = df_model[df_model.index.year == 2021] # 2021년 데이터
    df_train_index = [i for i in df_model.index if i not in df_test.index]
    df_train = df_model.loc[df_train_index]

    # autogluon 학습을 위한 데이터 형태로 변환
    train = TabularDataset(df_train)
    test = TabularDataset(df_test)

    print(train.head())

    predictor= TabularPredictor(label ='화재합계').fit(train_data = train, verbosity = 2,presets='best_quality')
    performance = predictor.evaluate(test)
    predictions = predictor.predict(test) # 새로운 데이터 셋에 대한 예측값 생성
    model = predictor.get_model_best() # 또는 predictor.model_type

    result_df = pd.DataFrame({'observation' : test['화재합계'],
                              'Predictions': predictions})
    
    csv_file_path = f'C:/빅데이터 활용 미래 사회문제 해결 아이디어 해커톤/모델결과_결과값/result_{file_name[i]}'
    result_df.to_csv(csv_file_path, index=False)   

    model_result = pd.DataFrame({'Performance': [performance],
                                 'Model': [model]})
    
    csv_file_path1 = f'C:/빅데이터 활용 미래 사회문제 해결 아이디어 해커톤/모델 결과_모델정보/model_{file_name[i]}'
    model_result.to_csv(csv_file_path1, index=False)   