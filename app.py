import os
import json
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

# Инициализируем FastAPI с метаданными для красивого отображения в Swagger UI
app = FastAPI(
    title="Mushroom Edibility Classifier API",
    description=(
        "REST-API для определения съедобности дикорастущих грибов по их "
        "морфологическим характеристикам. Модель оптимизирована под метрику Recall."
    ),
    version="1.0.0"
)

# Определение путей к файлам артефактов
MODEL_PATH = 'mushroom_model.pkl'
ENCODERS_PATH = 'label_encoders.pkl'
FEATURES_PATH = 'feature_names.json'

# Глобальные переменные для хранения загруженных объектов
model = None
encoders = None
features = None


@app.on_event("startup")
def startup_event():
    """Выполняется при запуске сервера. Гарантирует безопасную загрузку ML-артефактов."""
    global model, encoders, features
    
    # Проверяем физическое наличие файлов на сервере
    missing_files = [f for f in [MODEL_PATH, ENCODERS_PATH, FEATURES_PATH] if not os.path.exists(f)]
    if missing_files:
        raise RuntimeError(
            f"Критическая ошибка: отсутствуют файлы моделей: {missing_files}. "
            "Запустите jupyter-ноутбук для генерации артефактов."
        )
        
    try:
        model = joblib.load(MODEL_PATH)
        encoders = joblib.load(ENCODERS_PATH)
        with open(FEATURES_PATH, 'r') as f:
            features = json.load(f)
        print(">>> Все ML-артефакты успешно загружены.")
    except Exception as e:
        raise RuntimeError(f"Ошибка при инициализации моделей: {e}")


class MushroomFeatures(BaseModel):
    """Схема входных параметров гриба для валидации данных (Pydantic)."""
    cap_shape: str = Field("x", description="Форма шляпки (b=bell, c=conical, x=flat, ...)")
    cap_surface: str = Field("s", description="Поверхность шляпки (f=fibrous, s=smooth, ...)")
    cap_color: str = Field("n", description="Цвет шляпки (n=brown, b=buff, g=gray, ...)")
    bruises: str = Field("t", description="Наличие пятен/потемнений при надавливании (t=yes, f=no)")
    odor: str = Field("p", description="Запах (a=almond, l=anise, p=pungent, n=none, ...)")
    gill_attachment: str = Field("f", description="Прикрепление жабр")
    gill_spacing: str = Field("c", description="Расстояние между жабрами")
    gill_size: str = Field("n", description="Размер жабр (b=broad, n=narrow)")
    gill_color: str = Field("k", description="Цвет жабр")
    stalk_shape: str = Field("e", description="Форма ножки")
    stalk_root: str = Field("e", description="Корень ножки (включая '?' для пропущенных)")
    stalk_surface_above_ring: str = Field("s", description="Шероховатость ножки выше кольца")
    stalk_surface_below_ring: str = Field("s", description="Шероховатость ножки ниже кольца")
    stalk_color_above_ring: str = Field("w", description="Цвет ножки выше кольца")
    stalk_color_below_ring: str = Field("w", description="Цвет ножки ниже кольца")
    veil_type: str = Field("p", description="Тип покрывала")
    veil_color: str = Field("w", description="Цвет покрывала")
    ring_number: str = Field("o", description="Количество колец")
    ring_type: str = Field("p", description="Тип кольца")
    spore_print_color: str = Field("k", description="Цвет спорового порошка")
    population: str = Field("s", description="Популяция")
    habitat: str = Field("u", description="Среда обитания (g=grasses, u=urban, d=woods, ...)")

    class Config:
        # Пример запроса, который будет по умолчанию отображаться в Swagger UI
        schema_extra = {
            "example": {
                "cap_shape": "x",
                "cap_surface": "s",
                "cap_color": "n",
                "bruises": "t",
                "odor": "p",
                "gill_attachment": "f",
                "gill_spacing": "c",
                "gill_size": "n",
                "gill_color": "k",
                "stalk_shape": "e",
                "stalk_root": "e",
                "stalk_surface_above_ring": "s",
                "stalk_surface_below_ring": "s",
                "stalk_color_above_ring": "w",
                "stalk_color_below_ring": "w",
                "veil_type": "p",
                "veil_color": "w",
                "ring_number": "o",
                "ring_type": "p",
                "spore_print_color": "k",
                "population": "s",
                "habitat": "u"
            }
        }


@app.post("/predict", summary="Классифицировать гриб по характеристикам")
def predict(mushroom: MushroomFeatures):
    """
    Классифицирует съедобность гриба:
    
    - Принимает на вход морфологические характеристики в виде буквенных обозначений.
    - Трансформирует переменные в формат датасета (заменяет '_' на '-').
    - Выполняет Label Encoding на основе предобученных энкодеров.
    - Генерирует новые фичи (odor_spore_combo, is_no_odor).
    - Возвращает предсказанный класс и вероятности классификации.
    """
    if model is None or encoders is None or features is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Сервис не инициализирован. Модели не загружены на сервере."
        )

    # Приведение ключей к исходному виду датасета (замена "_" на "-")
    raw_dict = mushroom.dict()
    input_dict = {k.replace('_', '-'): v for k, v in raw_dict.items()}
    
    # Обработка неопределенного значения "stalk-root" — замена на моду, как в Jupyter-ноутбуке
    if input_dict.get("stalk-root") == "?":
        input_dict["stalk-root"] = "b"
        
    sample_df = pd.DataFrame([input_dict])
    
    # Кодирование категориальных признаков
    for col in sample_df.columns:
        if col in encoders:
            try:
                sample_df[col] = encoders[col].transform(sample_df[col])
            except ValueError:
                # В случае неизвестной категории (OOD) кодируем дефолтным значением 0
                sample_df[col] = 0
                
    # Feature Engineering (идентично этапу обучения)
    sample_df['odor_spore_combo'] = sample_df['odor'] * 10 + sample_df['spore-print-color']
    
    try:
        no_odor_val = encoders['odor'].transform(['n'])[0]
        sample_df['is_no_odor'] = (sample_df['odor'] == no_odor_val).astype(int)
    except Exception:
        sample_df['is_no_odor'] = 0
    
    # Сортировка колонок в строгом соответствии с обучающим набором
    try:
        sample_df = sample_df[features]
    except KeyError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Несоответствие структуры признаков: отсутствует {e}"
        )
    
    # Расчет предсказаний
    pred = int(model.predict(sample_df)[0])
    prob = model.predict_proba(sample_df)[0]
    
    return {
        "status": "success",
        "result": "Ядовитый ☠️" if pred == 1 else "Съедобный",
        "probabilities": {
            "edible": f"{prob[0]:.2%}",
            "poisonous": f"{prob[1]:.2%}"
        },
        "danger_alert": pred == 1
    }


@app.get("/", summary="Статус проверки API")
def root():
    """Эндпоинт проверки работоспособности сервиса."""
    return {
        "status": "active",
        "message": "Mushroom API is fully running. Please navigate to /docs for Swagger UI documentation."
    }
