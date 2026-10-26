# 1. 붓꽃(Iris) 데이터셋 로드 및 분석 설정

import pandas as pd, numpy as np, joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
import os

SEED = 42
RESULT_1 = 'submission_1'  # 제출 파일명 접두어

# 붓꽃 데이터 경로 설정
TRAIN_URL = 'https://github.com/gzone2000/TEMP_TEST/raw/refs/heads/master/iris.csv'
TEST_URL = None

df_all = pd.read_csv(TRAIN_URL)
df_test_raw = pd.read_csv(TEST_URL) if TEST_URL else None
print('불러온 데이터:', df_all.shape, '| test 파일:', df_test_raw.shape if TEST_URL else '없음(train 에서 분리)')
display(df_all.head())

# 2. 타겟 지정 및 데이터 분할 (variety 예측)
TARGET = 'variety'
PRED_COL = TARGET
DROP_COLS = []  # 붓꽃 데이터는 버릴 불필요 ID 성격의 고유 문자열 컬럼이 없습니다

print('타겟 분포:')
print(df_all[TARGET].value_counts())

if df_test_raw is None:
    df, df_test = train_test_split(df_all, test_size=0.2, random_state=SEED, stratify=df_all[TARGET])
else:
    df, df_test = df_all.copy(), df_test_raw.copy()

df, df_test = df.reset_index(drop=True), df_test.reset_index(drop=True)
df_test_orig = df_test.copy()
print('Train:', df.shape, '/ Test:', df_test.shape)

#3. 결측치 처리, 인코딩 및 스케일링 전처리
has_y_test = TARGET in df_test.columns
X      = df.drop(columns=[c for c in DROP_COLS + [TARGET] if c in df.columns], errors='ignore')
X_test = df_test.drop(columns=[c for c in DROP_COLS + [TARGET] if c in df_test.columns], errors='ignore')

num = X.select_dtypes('number').columns.tolist()
cat = [c for c in X.columns if c not in num]

# 결측 처리
if num:
    imp_n = SimpleImputer(strategy='median')
    X[num] = imp_n.fit_transform(X[num])
    X_test[num] = imp_n.transform(X_test[num])
if cat:
    imp_c = SimpleImputer(strategy='most_frequent')
    X[cat] = imp_c.fit_transform(X[cat])
    X_test[cat] = imp_c.transform(X_test[cat])

# 범주형 변수 변환
for col in cat:
    le = LabelEncoder().fit(X[col].astype(str))
    m = {v: i for i, v in enumerate(le.classes_)}
    X[col] = X[col].astype(str).map(m)
    X_test[col] = X_test[col].astype(str).map(m).fillna(-1).astype(int)

# 타겟 변수 라벨인코딩
ty = LabelEncoder()
y = ty.fit_transform(df[TARGET])
y_test = ty.transform(df_test[TARGET]) if has_y_test else None

# 스케일링
sc = StandardScaler()
X = pd.DataFrame(sc.fit_transform(X), columns=X.columns)
X_test = pd.DataFrame(sc.transform(X_test), columns=X_test.columns)
print('수치형:', len(num), '· 범주형:', len(cat), '· 클래스:', list(ty.classes_))

#4. 모델 교차검증 및 학습
Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)

models = {
    'LogReg':       LogisticRegression(max_iter=1000, random_state=SEED),
    'RandomForest': RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1),
    'GradBoost':    GradientBoostingClassifier(random_state=SEED),
}

cv = {n: cross_val_score(m, Xtr, ytr, cv=5).mean() for n, m in models.items()}
best_name = max(cv, key=cv.get)
best = models[best_name].fit(Xtr, ytr)
print('CV Scores:', {n: round(s, 4) for n, s in cv.items()}, '→ Best:', best_name)

yva_pred = best.predict(Xva)
print(f"Val Acc {accuracy_score(yva, yva_pred):.4f} / F1(macro) {f1_score(yva, yva_pred, average='macro'):.4f}")
print(classification_report(yva, yva_pred, target_names=[str(c) for c in ty.classes_]))

#5. 예측 및 결과 파일 저장
pred = ty.inverse_transform(best.predict(X_test))

sub = df_test_orig.drop(columns=[c for c in [TARGET] if c in df_test_orig.columns]).copy()
sub[PRED_COL] = pred

print('행수', len(sub), '| 컬럼', list(sub.columns))
print('예측 분포:', sub[PRED_COL].value_counts().to_dict())
if has_y_test:
    print('Test Acc', round(accuracy_score(y_test, best.predict(X_test)), 4))

CSV, MODEL = f'{RESULT_1}.csv', f'{RESULT_1}.pkl'
assert len(sub) == len(df_test_orig), '행 수 불일치'
assert sub[PRED_COL].isnull().sum() == 0, '예측 결측'
sub.to_csv(CSV, index=False)
joblib.dump(best, MODEL)
print('저장 완료:', [f for f in os.listdir('.') if f.startswith(RESULT_1)])





import pandas as pd, numpy as np, joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

SEED = 42
RESULT_1 = 'submission_1'             # 제출 파일명 접두어. Tabular=1

# 1. 설정 · 로드 ← ★ 여기만 수정
# ▼ 데이터 선택 (하나만) — 회귀 예제는 전부 csv(콤마 구분)
TRAIN_URL = 'https://github.com/gzone2000/TEMP_TEST/raw/refs/heads/master/kc_house_data.csv'   # [1] 집값
# TRAIN_URL = 'https://github.com/gzone2000/TEMP_TEST/raw/refs/heads/master/ade_sales.csv'      # [2] 레모네이드
TEST_URL = None                        # test 파일 따로 있으면 URL, 없으면 None(train 에서 분리)

df_all = pd.read_csv(TRAIN_URL)                     # 원본(불변) — 분할은 타겟 정한 뒤 '3.' 에서
df_test_raw = pd.read_csv(TEST_URL) if TEST_URL else None
print('불러온 데이터:', df_all.shape, '| test 파일:', df_test_raw.shape if TEST_URL else '없음(train 에서 분리)')
df_all.head()

# 2. 데이터 확인 (컬럼·타입·결측을 보고 아래에서 맞출/버릴 컬럼을 정한다)
print(df_all.dtypes)
# 고유값이 행수에 가까운 컬럼 = ID/날짜 → 버릴 후보
print(); print('컬럼별 고유값 수:'); print(df_all.nunique().sort_values(ascending=False))
print(); print('결측:', df_all.isnull().sum()[lambda s: s > 0].to_dict())


# 3. 맞출 컬럼 지정 · 분할 ← ★ 위 '데이터 확인'을 보고 채운다
# ★ 위 컬럼 목록에서 '맞출 컬럼(예측할 수치)'을 직접 지정 (지문이 준 이름과 철자 일치 확인)
TARGET = 'price'             # [1] 집값 · [2] 레모네이드:'Lemon'
PRED_COL = TARGET            # 제출 결과 컬럼명 — 지문이 다른 이름을 지정하면 그 이름으로 바꾼다

print('타겟 통계:'); print(df_all[TARGET].describe()[['mean', 'min', 'max']])

# test 가 따로 없으면 여기서 분리 (회귀는 stratify 없음). df_all 은 원본이라 재실행해도 안전
if df_test_raw is None:
    df, df_test = train_test_split(df_all, test_size=0.2, random_state=SEED)
else:
    df, df_test = df_all.copy(), df_test_raw.copy()
df, df_test = df.reset_index(drop=True), df_test.reset_index(drop=True)
df_test_orig = df_test.copy()          # 제출은 test 원본 구조로
print('Train', df.shape, '/ Test', df_test.shape)


# 4. 상관 히트맵 (숫자 컬럼 ↔ 타겟)
import matplotlib.pyplot as plt
num = df.select_dtypes('number').copy()
if TARGET not in num.columns:                       # 문자 타겟이면 상관 계산 위해 임시 인코딩
    num[TARGET] = LabelEncoder().fit_transform(df[TARGET].astype(str))
corr = num.corr(numeric_only=True)

# 타겟과 상관 큰 순 (|r|=1 이면 누수 의심, 클수록 강한 예측변수)
tcorr = corr[TARGET].drop(TARGET).abs().sort_values(ascending=False)
print('타겟과 상관(|r|) 상위:'); print(tcorr.head(10).round(3).to_dict())

# 컬럼이 많으면(>15) 타겟 + 상관 상위 12개만 그린다 (전체는 너무 커서 안 보임)
cols = ([TARGET] + tcorr.head(12).index.tolist()) if len(corr) > 15 else corr.columns.tolist()
cm = corr.loc[cols, cols]
plt.figure(figsize=(min(1 + 0.55 * len(cols), 11), min(1 + 0.55 * len(cols), 11)))
plt.imshow(cm, cmap='coolwarm', vmin=-1, vmax=1); plt.colorbar(fraction=0.046)
plt.xticks(range(len(cols)), cols, rotation=90); plt.yticks(range(len(cols)), cols)
for i in range(len(cols)):
    for j in range(len(cols)):
        plt.text(j, i, f'{cm.iloc[i, j]:.2f}', ha='center', va='center', fontsize=7)
plt.title('Correlation heatmap (numeric + target)'); plt.tight_layout(); plt.show()


# 5. 버릴 컬럼 결정 ← ★ 위 '데이터 확인'을 보고 채운다
# ① 고유값/행수 비율(1에 가까울수록 ID·날짜 후보):
#   {'id': 0.994, 'sqft_lot': 0.486, 'sqft_lot15': 0.437, 'lat': 0.28, 'sqft_living': 0.053, 'sqft_above': 0.049}
#   → 비율 0.9 초과: ['id']
#② 타겟과 완벽히 일치(누수 의심): 없음
#버릴 컬럼: ['id', 'date']
#남는 피처: ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade', 'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode', 'lat', 'long', 'sqft_living15', 'sqft_lot15']
DROP_COLS = ['PassengerId', 'Name']   

# 6. 전처리 (결측 → 인코딩 → 스케일)
has_y_test = TARGET in df_test.columns
X      = df.drop(columns=[c for c in DROP_COLS + [TARGET] if c in df.columns], errors='ignore')
X_test = df_test.drop(columns=[c for c in DROP_COLS + [TARGET] if c in df_test.columns], errors='ignore')
y      = df[TARGET].astype(float)                      # 회귀 타겟(연속값) — 인코딩 안 함
y_test = df_test[TARGET].astype(float) if has_y_test else None

# 수치형/범주형 분리 — 결측 채우는 방법이 다르므로 먼저 나눈다
num = X.select_dtypes('number').columns.tolist()
cat = [c for c in X.columns if c not in num]

# 결측치 채우기 — ★ train 에서 fit → test 는 transform (test 로 fit 하면 누수)
if num:
    imp_n = SimpleImputer(strategy='median')           # 수치형: 중앙값
    X[num] = imp_n.fit_transform(X[num]); X_test[num] = imp_n.transform(X_test[num])
if cat:
    imp_c = SimpleImputer(strategy='most_frequent')    # 범주형: 최빈값
    X[cat] = imp_c.fit_transform(X[cat]); X_test[cat] = imp_c.transform(X_test[cat])

# 범주형 인코딩 (train 사전으로 test 변환, 미지값은 -1)
for col in cat:
    le = LabelEncoder().fit(X[col].astype(str)); m = {v: i for i, v in enumerate(le.classes_)}
    X[col] = X[col].astype(str).map(m)
    X_test[col] = X_test[col].astype(str).map(m).fillna(-1).astype(int)

# 스케일링
sc = StandardScaler()
X = pd.DataFrame(sc.fit_transform(X), columns=X.columns)
X_test = pd.DataFrame(sc.transform(X_test), columns=X_test.columns)
print('수치형', len(num), '· 범주형', len(cat))


# 7. 모델 비교 · 학습 · 평가 (RMSE · MAE · R²)
def rmse(a, b): return np.sqrt(mean_squared_error(a, b))
Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.2, random_state=SEED)

models = {
    'Linear':       LinearRegression(),
    'RandomForest': RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=-1),
    'GradBoost':    GradientBoostingRegressor(random_state=SEED),
}
cv = {n: cross_val_score(m, Xtr, ytr, cv=5, scoring='r2').mean() for n, m in models.items()}
best_name = max(cv, key=cv.get)
best = models[best_name].fit(Xtr, ytr)
print('CV R²:', {n: round(s, 4) for n, s in cv.items()}, '→ 1위', best_name)

pv = best.predict(Xva)
print(f'Val RMSE {rmse(yva, pv):.3f} / MAE {mean_absolute_error(yva, pv):.3f} / R² {r2_score(yva, pv):.4f}')


# 8. 결과 만들고 ⭐ 눈으로 확인
pred = best.predict(X_test)

sub = df_test_orig.drop(columns=[c for c in [TARGET] if c in df_test_orig.columns]).copy()
sub[PRED_COL] = pred

print('행수', len(sub), '| 컬럼', list(sub.columns))
print('예측 요약:', {'min': round(float(pred.min()), 1), 'mean': round(float(pred.mean()), 1), 'max': round(float(pred.max()), 1)})
if has_y_test:
    print('Test RMSE', round(rmse(y_test, pred), 3), '/ R²', round(r2_score(y_test, pred), 4))
sub.head(10)


# 9. ⭐ csv · 모델 저장
import os
CSV, MODEL = f'{RESULT_1}.csv', f'{RESULT_1}.pkl'
assert len(sub) == len(df_test_orig), '행 수 불일치'      # ★ 저장 전에 검증 (0점 나는 두 원인)
assert sub[PRED_COL].isnull().sum() == 0, '예측 결측'
sub.to_csv(CSV, index=False)     # 결과 csv
joblib.dump(best, MODEL)                                # 모델
print('저장 완료:', [f for f in os.listdir('.') if f.startswith(RESULT_1)])






import pandas as pd, numpy as np, re, joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score, classification_report

# 1. 설정 · 로드 ← ★ 여기만 수정
SEED = 42
RESULT_2 = 'submission_2'             # 제출 파일명 접두어. Text=2

# ▼ 데이터 선택 (하나만) — '파일 여는 것'만. 문장/라벨 컬럼은 데이터 본 뒤 '3.' 에서
# TRAIN_URL, TEST_URL = 'https://github.com/gzone2000/TEMP_TEST/raw/refs/heads/master/A_comment_train.xlsx', 'https://github.com/gzone2000/TEMP_TEST/raw/refs/heads/master/A_comment_test.xlsx'  # [1] A댓글·이진·train/test 분리
TRAIN_URL, TEST_URL = 'https://github.com/gzone2000/TEMP_TEST/raw/refs/heads/master/ChatbotData.csv', None       # [2] 챗봇·다중(3)
# TRAIN_URL, TEST_URL = 'https://github.com/gzone2000/TEMP_TEST/raw/refs/heads/master/IMDb_small_data.csv', None    # [3] IMDb·이진(영어)

def read_any(u):
    return pd.read_excel(u) if str(u).lower().endswith(('.xlsx', '.xls')) else pd.read_csv(u)

df_all = read_any(TRAIN_URL)                        # 원본(불변) — 분할은 컬럼 정한 뒤 '3.' 에서
df_test_raw = read_any(TEST_URL) if TEST_URL else None
for _d in [df_all] + ([df_test_raw] if TEST_URL else []):   # 저장 시 생긴 인덱스 컬럼 제거
    _d.drop(columns=[c for c in _d.columns if str(c).startswith('Unnamed')], inplace=True, errors='ignore')
print('불러온 데이터:', df_all.shape, '| test 파일:', df_test_raw.shape if TEST_URL else '없음(train 에서 분리)')
df_all.head()

#2. 데이터 확인 (어느 컬럼이 문장이고 어느 컬럼이 라벨인지 본다)
print('컬럼:', list(df_all.columns))
print(); print('행 예시:'); print(df_all.head(3))

#3. 문장 · 라벨 컬럼 지정 · 분할 ← ★ 위 '데이터 확인'을 보고 채운다
# ★ 위 컬럼에서 '문장 컬럼(TEXT_COL)'과 '라벨 컬럼(TARGET)'을 직접 지정
TEXT_COL = 'Q'            # [1] A댓글:'data' · [2] 챗봇:'Q' · [3] IMDb:'review'
TARGET   = 'label'           # [1] 'label' · [2] 'label' · [3] 'sentiment'
PRED_COL = TARGET            # 제출 결과 컬럼명 — 지문이 다른 이름을 지정하면 그 이름으로

print('텍스트 예시:', df_all[TEXT_COL].iloc[0])
print('label 고유값:', sorted(df_all[TARGET].dropna().unique().tolist(), key=str))   # 숫자면 '4.' 에서 매핑
print(df_all[TARGET].value_counts())

# test 가 따로 없으면 여기서 분리 (stratify=라벨). df_all 은 원본이라 재실행해도 안전
if df_test_raw is None:
    df, df_test = train_test_split(df_all, test_size=0.2, random_state=SEED, stratify=df_all[TARGET])
else:
    df, df_test = df_all.copy(), df_test_raw.copy()
df, df_test = df.reset_index(drop=True), df_test.reset_index(drop=True)
df_test_orig = df_test.copy()          # 제출은 test 원본 구조로
print('Train', df.shape, '/ Test', df_test.shape)

#4. 라벨 값 수정 ← ★ 숫자일 때만 채운다
# 제출값 = label 그대로. 라벨이 '이미 문자'면 None, '숫자'면 의미대로 매핑.
# LABEL_MAP = None
LABEL_MAP = {0: '일상', 1: '부정', 2: '긍정'} # 예) 챗봇(0/1/2):
# 예) IMDb(0/1):   LABEL_MAP = {0: 'negative', 1: 'positive'}

if LABEL_MAP:
    df[TARGET] = df[TARGET].map(LABEL_MAP)
    if TARGET in df_test.columns:
        df_test[TARGET] = df_test[TARGET].map(LABEL_MAP)
print('최종 label:', sorted(df[TARGET].unique().tolist(), key=str))

#5. 전처리 (정제 → train/val 분할)
def clean(t):
    t = re.sub('[^가-힣a-zA-Z0-9 ]', ' ', str(t))       # 특수문자 제거
    return ' '.join(t.split())                         # 다중 공백 정리

df[TEXT_COL] = df[TEXT_COL].map(clean)
df_test[TEXT_COL] = df_test[TEXT_COL].map(clean)
df = df[df[TEXT_COL] != ''].reset_index(drop=True)

has_y_test = TARGET in df_test.columns
Xtr, Xva, ytr, yva = train_test_split(
    df[TEXT_COL], df[TARGET], test_size=0.2, random_state=SEED, stratify=df[TARGET])
print('train', len(Xtr), '/ val', len(Xva))

#6. TF-IDF · 모델 비교 · 평가
# 한국어 짧은 문장엔 문자 n-gram(char_wb)이 공백 토큰화보다 강하다
vec = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4), max_features=50000, min_df=2, sublinear_tf=True)
Xtr_v = vec.fit_transform(Xtr); Xva_v = vec.transform(Xva)

models = {'LinearSVC': LinearSVC(C=1), 'LogReg': LogisticRegression(max_iter=1000), 'NB': MultinomialNB(alpha=0.1)}
cv = {n: cross_val_score(m, Xtr_v, ytr, cv=5).mean() for n, m in models.items()}
best_name = max(cv, key=cv.get)
best = models[best_name].fit(Xtr_v, ytr)
print('CV:', {n: round(s, 4) for n, s in cv.items()}, '→ 1위', best_name)

yva_pred = best.predict(Xva_v)
print(f'Val Acc {accuracy_score(yva, yva_pred):.4f} / F1(macro) {f1_score(yva, yva_pred, average="macro"):.4f}')
print(classification_report(yva, yva_pred))

#7. 결과 만들고 ⭐ 눈으로 확인
pred = best.predict(vec.transform(df_test[TEXT_COL]))

sub = df_test_orig.drop(columns=[c for c in [TARGET] if c in df_test_orig.columns]).copy()
sub[PRED_COL] = pred

rng = np.random.RandomState(SEED)                       # ★ 원문 vs 예측 눈 확인
for i in rng.choice(len(sub), min(8, len(sub)), replace=False):
    print(f'[{sub[PRED_COL].iloc[i]}]  {str(df_test_orig[TEXT_COL].iloc[i])[:50]}')
if has_y_test:
    print('\nTest Acc', round(accuracy_score(df_test[TARGET], pred), 4))
sub.head()

#8. ⭐ csv · 모델 저장
import os
CSV, MODEL = f'{RESULT_2}.csv', f'{RESULT_2}.pkl'
assert len(sub) == len(df_test_orig), '행 수 불일치'      # ★ 저장 전에 검증 (0점 나는 두 원인)
assert sub[PRED_COL].isnull().sum() == 0, '예측 결측'
sub.to_csv(CSV, index=False)      # 결과 csv
joblib.dump({'model': best, 'vectorizer': vec}, MODEL)  # 모델 + 벡터라이저
print('저장 완료:', [f for f in os.listdir('.') if f.startswith(RESULT_2)])





import tensorflow as tf, pathlib, os, glob, numpy as np, pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report

SEED = 42; tf.random.set_seed(SEED)

# 1. 설정 · 예제 데이터 내려받기

RESULT_3 = 'submission_3'             # 제출 파일명 접두어. Image=3
PRED_COL = 'pred'                      # 결과 컬럼명 (지문이 지정하면 그 이름)
IMAGE_SIZE, BATCH = (224, 224), 32
EPOCHS_1, EPOCHS_2 = 8, 5              # Phase1(동결) · Phase2(미세조정)

# ▼ (연습용) 예제 데이터 내려받기 — 시험에선 데이터가 이미 주어지므로 이 셀은 건너뜀
DATA_URL = 'https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz'  # 꽃 5클래스
# DATA_URL = 'https://storage.googleapis.com/download.tensorflow.org/data/rps.zip'  # 가위바위보 3클래스
try:
    tf.keras.utils.get_file(origin=DATA_URL, extract=True, cache_dir='.', cache_subdir='data')  # 현재 폴더 ./data 에 해제
except Exception as e:
    print('다운로드/해제 건너뜀:', type(e).__name__)

# ▼ 어디에 풀렸는지 눈으로 확인 → 이미지가 들어 있는 폴더들의 '바로 위'가 data_dir
_ds = pathlib.Path('data')
print("'./data' 안 이미지 폴더 (= 카테고리):")
for _p in sorted(d for d in _ds.rglob('*') if d.is_dir()):
    _n = sum(len(glob.glob(str(_p / e))) for e in ('*.jpg', '*.jpeg', '*.png'))
    if _n:
        print(f'   {_p}   · 이미지 {_n}장')

# 1b. train / test 경로 지정 ← ★ 위 목록을 보고 직접
# ★ 위 목록에서 이미지 폴더들의 '바로 위' 폴더를 직접 지정
# ★★★ 이미지는 여기서 디렉토리 경로 잡는게 제일 중요 ★★★
data_dir = pathlib.Path('data') / 'flower_photos.tgz' / 'flower_photos'   # rps 면 'data/rps' / 시험: 주어진 train 폴더
# data_dir = pathlib.Path('/content/train')          # 예) train 폴더가 여기 있으면

TEST_DIR = None                        # 예: './test' (라벨 없는 test 이미지 폴더). 없으면 train 의 val 로 데모

assert data_dir.exists(), f'{data_dir} 없음 — 위 목록을 보고 실제 경로로 고치세요'
print('train 경로:', data_dir, '| 하위:', [d.name for d in data_dir.iterdir() if d.is_dir()])

# 2. 데이터 확인 (클래스별 장수)
for c in sorted(d.name for d in data_dir.iterdir() if d.is_dir()):
    n = sum(len(list((data_dir / c).glob(e))) for e in ('*.jpg', '*.jpeg', '*.png'))
    print(f'  {c}: {n}장')

# 3. 데이터셋 생성 (train/val 같은 seed 로 분할)
com = dict(label_mode='categorical', image_size=IMAGE_SIZE, batch_size=BATCH, seed=SEED)
# train/val 둘 다 shuffle=True + 같은 seed → 클래스가 고루 섞여 분할된다.
#   (val 에 shuffle=False 를 쓰면 파일이 클래스 폴더 순서라 마지막 클래스만 val 로 몰림!)
train_ds = tf.keras.utils.image_dataset_from_directory(data_dir, validation_split=0.2, subset='training', shuffle=True, **com)
val_ds   = tf.keras.utils.image_dataset_from_directory(data_dir, validation_split=0.2, subset='validation', shuffle=True, **com)
class_names = train_ds.class_names
NUM_CLASSES = len(class_names)
val_files = list(val_ds.file_paths)                    # val 파일 목록(고정 순서) — 평가/제출 정렬용
train_ds = train_ds.prefetch(tf.data.AUTOTUNE); val_ds = val_ds.prefetch(tf.data.AUTOTUNE)
print('클래스', NUM_CLASSES, class_names)

# 4. 모델 · 학습 (MobileNetV2 · 2단계)
base = tf.keras.applications.MobileNetV2(input_shape=IMAGE_SIZE + (3,), include_top=False, weights='imagenet')
base.trainable = False
inp = tf.keras.Input(IMAGE_SIZE + (3,))
x = tf.keras.layers.Rescaling(1 / 127.5, offset=-1)(inp)   # MobileNetV2 는 -1~1 입력
x = base(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.3)(x)
out = tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')(x)   # 이진/다중 통일
model = tf.keras.Model(inp, out)

CKPT = f'{RESULT_3}_ckpt.weights.h5'
cbs = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True),
       tf.keras.callbacks.ModelCheckpoint(CKPT, save_best_only=True, save_weights_only=True, monitor='val_loss')]
# ↑ 두 단계가 이 콜백을 공유 → val_loss 최저 가중치가 Phase 1·2 통틀어 CKPT 에 유지됨

# Phase 1: base 동결, 분류 헤드만 (lr 1e-3)
model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_1, callbacks=cbs, verbose=1)

# Phase 1 이 개선 없어 조기 종료돼도, 여기부터 그대로 이어서 파인튜닝(fit 이 따로라 patience 새로 시작).
# Phase 2: 상위 30층 해제, 낮은 lr (1e-5) — 높으면 사전학습 가중치가 망가짐
base.trainable = True
for l in base.layers[:-30]:
    l.trainable = False
model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_2, callbacks=cbs, verbose=1)

# ★ 학습이 개선 못 하거나 조기 중단돼도 CKPT 엔 두 단계 통틀어 best 가 있다 → 그걸 복원 (버전 무관)
model.load_weights(CKPT)

# 5. 평가 · ⭐ 눈으로 확인 (val)
# 파일 경로로 직접 로드 → image_dataset 의 셔플과 무관하게 파일-예측 정렬 보장
def load_img(path):
    img = tf.image.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
    return tf.image.resize(img, IMAGE_SIZE)
def predict_files(files):
    ds = tf.data.Dataset.from_tensor_slices([str(f) for f in files]).map(load_img).batch(BATCH).prefetch(tf.data.AUTOTUNE)
    return model.predict(ds, verbose=0).argmax(1)

val_pred = predict_files(val_files)
val_true = np.array([class_names.index(pathlib.Path(f).parent.name) for f in val_files])   # 폴더명 = 정답
print('Val Acc', round(accuracy_score(val_true, val_pred), 4))
print(classification_report(val_true, val_pred, target_names=class_names))

# ⭐ 눈으로: 초록=정답 / 빨강=오답
rng = np.random.RandomState(SEED)
plt.figure(figsize=(12, 6))
for j, i in enumerate(rng.choice(len(val_files), min(8, len(val_files)), replace=False)):
    plt.subplot(2, 4, j + 1); plt.imshow(tf.keras.utils.load_img(val_files[i])); plt.axis('off')
    ok = val_pred[i] == val_true[i]
    plt.title(f'P:{class_names[val_pred[i]]}', color='green' if ok else 'red', fontsize=9)
plt.tight_layout(); plt.show()

# 6. 결과 만들기 (test 폴더가 있으면 그걸로, 없으면 val 데모)
if TEST_DIR:
    # 시험: 라벨 없는 test 폴더의 이미지들 (순서 고정 = 파일명 정렬)
    test_files = sorted(f for f in glob.glob(os.path.join(TEST_DIR, '**', '*.*'), recursive=True)
                        if f.lower().endswith(('.jpg', '.jpeg', '.png')))
    assert test_files, f'{TEST_DIR} 에 이미지가 없습니다'
    sub_files = test_files
    sub_pred = predict_files(test_files)
else:
    sub_files, sub_pred = val_files, val_pred          # 데모: 위 val 예측 재사용

sub = pd.DataFrame({'filename': [os.path.basename(f) for f in sub_files],
                    PRED_COL: [class_names[i] for i in sub_pred]})
print('행수', len(sub), '| 예측 분포:', dict(pd.Series(sub[PRED_COL]).value_counts()))
sub.head(10)

# 7. ⭐ csv · 모델 저장
CSV, MODEL = f'{RESULT_3}.csv', f'{RESULT_3}.keras'
assert sub[PRED_COL].isnull().sum() == 0, '예측 결측'     # ★ 저장 전에 검증
sub.to_csv(CSV, index=False)
model.save(MODEL)
print('저장 완료:', [f for f in os.listdir('.') if f.startswith(RESULT_3)])
