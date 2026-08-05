import os
import io
import base64
from contextlib import asynccontextmanager
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import urllib.request
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

from fastapi import FastAPI, HTTPException, Request, Form
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates


models = {}
day_profile = None
df_historical = None
MODEL_URLS = {
    "reg_temp_rf_model.pkl": "https://google.com", 
    "reg_precip_rf_model.pkl": "https://google.com", 
    "clf_anomaly_rf_model.pkl": "https://google.com"
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Управляет жизненным циклом приложения и принудительно качает файлы моделей в облаке."""
    global day_profile, df_historical
    try:
        # Настройка кастомного сетевого агента, чтобы Google не блокировал частые запросы сервера
        opener = urllib.request.build_opener()
        opener.addheaders = [('User-agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)')]
        urllib.request.install_opener(opener)

        for model_name, url in MODEL_URLS.items():
            # Если файл модели поврежден или весит подозрительно мало (меньше 10 КБ - значит скачался HTML-огрызок)
            if not os.path.exists(model_name) or os.path.getsize(model_name) < 10240:
                print(f"📥 Принудительное скачивание тяжелой модели: {model_name}...")
                if os.path.exists(model_name):
                    os.remove(model_name) # Удаляем старый битый файл, если он был
                
                urllib.request.urlretrieve(url, model_name)
                print(f"✅ Файл {model_name} успешно скачан на диск хостинга. Размер: {os.path.getsize(model_name)} байт.")

        # Загрузка весов моделей Scikit-learn в оперативную память
        print("⚙️ Инициализация моделей в оперативную память...")
        models["reg_temp"] = joblib.load("reg_temp_rf_model.pkl")
        models["reg_precip"] = joblib.load("reg_precip_rf_model.pkl")
        models["clf_anomaly"] = joblib.load("clf_anomaly_rf_model.pkl")
        
        # Загрузка исторического климатического датасета
        df_historical = pd.read_csv("tashkent_climate_features_ready.csv")
        df_historical["Date"] = pd.to_datetime(df_historical["Date"])
        
        # Расчет профиля дней года для инференса
        day_profile = (
            df_historical.groupby("DayOfYear")[["Humidity", "Wind_Speed"]]
            .median()
            .reset_index()
        )
        print("🚀 Все модели и климатические профили успешно загружены в память сервера!")
    except Exception as e:
        print(f"❌ Критическая ошибка при инициализации lifespan: {str(e)}")
    yield
    models.clear()

app = FastAPI(title="Tashkent Climate Predictor", lifespan=lifespan)
templates = Jinja2Templates(directory="templates")

def generate_accurate_summer_forecast(df_historical, reg_temp, reg_precip, clf_anomaly, day_profile,
                                      days_to_forecast=30):

    forecast_records = []

    history = df_historical.sort_values('Date').tail(35).copy()
    current_date = pd.to_datetime(history['Date'].max())
    feature_columns = reg_temp.feature_names_in_

    for i in range(days_to_forecast):
        current_date = current_date + pd.Timedelta(days=1)
        doy = int(current_date.dayofyear)

        typical_weather = day_profile[day_profile['DayOfYear'] == doy]
        if not typical_weather.empty:
            humidity = float(typical_weather['Humidity'].values.item())
            wind = float(typical_weather['Wind_Speed'].values.item())
        else:
            humidity = float(history['Humidity'].iloc[-1])
            wind = float(history['Wind_Speed'].iloc[-1])

        new_row = {
            'Date': current_date,
            'Year': int(current_date.year),
            'Month': int(current_date.month),
            'DayOfYear': doy,
            'Humidity': humidity,
            'Wind_Speed': wind,
        }

        temp_col = "Temp_Max" if "Temp_Max" in history.columns else "Temp_Average"

        new_row['Temp_Lag_1'] = float(history[temp_col].iloc[-1])
        new_row['Temp_Lag_7'] = float(history[temp_col].iloc[-7])
        new_row['Temp_Lag_30'] = float(history[temp_col].iloc[-30])

        new_row['Precip_Lag_1'] = float(history['Precipitation'].iloc[-1])
        new_row['Precip_Lag_7'] = float(history['Precipitation'].iloc[-7])
        new_row['Precip_Lag_30'] = float(history['Precipitation'].iloc[-30])

        new_row['Temp_RollMean_30'] = float(history[temp_col].tail(30).mean())
        new_row['Precip_RollMean_30'] = float(history['Precipitation'].tail(30).mean())


        input_df = pd.DataFrame([new_row])[feature_columns]

        pred_t = float(reg_temp.predict(input_df).item())
        pred_p = float(max(0.0, reg_precip.predict(input_df).item()))


        new_row[temp_col] = pred_t
        new_row['Precipitation'] = pred_p


        is_anomaly = int(clf_anomaly.predict(input_df).item())

        proba_matrix = clf_anomaly.predict_proba(input_df)
        anomaly_proba = float(proba_matrix[0][1])


        history = pd.concat([history, pd.DataFrame([new_row])], ignore_index=True)

        forecast_records.append({
            'date': current_date.strftime("%Y-%m-%d"),
            'display_date': current_date.strftime("%d.%m"),
            'temp_max': round(pred_t, 1),
            'precip': round(pred_p, 2),
            'is_anomaly': is_anomaly,
            'probability': anomaly_proba
        })

    return forecast_records


@app.get("/api/download_report")
async def download_report(days: int = 30):
    if not models or df_historical is None:
        raise HTTPException(status_code=503, detail="Модели не загружены.")

    try:
        font_path = os.path.join(os.environ.get('WINDIR', 'C:\\Windows'), 'Fonts', 'arial.ttf')
        font_bd_path = os.path.join(os.environ.get('WINDIR', 'C:\\Windows'), 'Fonts', 'arialbd.ttf')

        pdfmetrics.registerFont(TTFont('Arial-Regular', font_path))
        pdfmetrics.registerFont(TTFont('Arial-Bold', font_bd_path))

        forecast_data = generate_accurate_summer_forecast(
            df_historical=df_historical,
            reg_temp=models["reg_temp"],
            reg_precip=models["reg_precip"],
            clf_anomaly=models["clf_anomaly"],
            day_profile=day_profile,
            days_to_forecast=days
        )

        anomalies_count = sum(1 for item in forecast_data if item["is_anomaly"] == 1)

        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
        story = []

        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'TitleStyle',
            fontName='Arial-Bold',
            fontSize=16,
            spaceAfter=15,
            textColor=colors.HexColor('#1a365d'),
            alignment=1
        )
        subtitle_style = ParagraphStyle('SubTitleStyle', fontName='Arial-Regular', fontSize=10, textColor=colors.gray,
                                        spaceAfter=20, alignment=1)
        h2_style = ParagraphStyle('H2Style', fontName='Arial-Bold', fontSize=13, spaceBefore=15, spaceAfter=10,
                                  textColor=colors.HexColor('#2c5282'))
        text_style = ParagraphStyle('TextStyle', fontName='Arial-Regular', fontSize=11, leading=15, spaceAfter=8)

        story.append(Paragraph("ОТЧЕТ ПО РЕЗУЛЬТАТАМ КЛИМАТИЧЕСКОГО ИНФЕРЕНСА", title_style))
        story.append(Paragraph(f"Горизонт прогнозирования: {days} дней. Город: Ташкент.", subtitle_style))
        story.append(Spacer(1, 10))

        story.append(Paragraph("1. Сводные результаты симуляции", h2_style))
        summary_text = f"За расчетный период в <font fontName='Arial-Bold'>{days} дней</font> математическими моделями Случайного Леса (Random Forest) было идентифицировано <font fontName='Arial-Bold'>{anomalies_count}</font> критических климатических аномалий (волн экстремальной жары или засушливых периодов)."
        story.append(Paragraph(summary_text, text_style))
        story.append(Spacer(1, 10))

        story.append(Paragraph("2. Статистические метрики точности прогнозных ядер", h2_style))

        def wrap_p(txt, is_bold=False):
            f_name = 'Arial-Bold' if is_bold else 'Arial-Regular'
            return Paragraph(f"<font fontName='{f_name}'>{txt}</font>", styles['Normal'])

        metrics_data = [
            [wrap_p('Целевой показатель', True), wrap_p('Метрика MAE', True), wrap_p('Метрика R-squared', True),
             wrap_p('Качественная оценка', True)],
            [wrap_p('Макс. температура'), wrap_p('1.65 °C'), wrap_p('0.9681'), wrap_p('Высокое качество (96.8%)')],
            [wrap_p('Атмосферные осадки'), wrap_p('0.97 мм'), wrap_p('0.4307'), wrap_p('Адекватная метео-норма')],
            [wrap_p('Классификатор рисков'), wrap_p('ROC-AUC: 0.9969'), wrap_p('Precision: 0.96'),
             wrap_p('Recall: 0.86')]
        ]

        t = Table(metrics_data, colWidths=[130, 100, 110, 180])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2b6cb0')),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f7fafc')),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.lightgrey),
        ]))
        story.append(t)

        doc.build(story)
        buffer.seek(0)

        return StreamingResponse(
            buffer,
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename=Climate_Report_{days}d.pdf"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка генерации PDF: {str(e)}")

@app.get("/", response_class=HTMLResponse)
async def get_dashboard(request: Request):
    if not models or df_historical is None:
        raise HTTPException(status_code=503, detail="Модели не загружены на сервере.")
    return templates.TemplateResponse(request, "index.html", context={})


@app.post("/predict", response_class=HTMLResponse)
async def post_predict(request: Request, daysParam: int = Form(...)):
    if not models or df_historical is None:
        raise HTTPException(status_code=503, detail="Модели не загружены на сервере.")

    try:

        forecast_data = generate_accurate_summer_forecast(
            df_historical=df_historical,
            reg_temp=models["reg_temp"],
            reg_precip=models["reg_precip"],
            clf_anomaly=models["clf_anomaly"],
            day_profile=day_profile,
            days_to_forecast=daysParam
        )

        df_forecast = pd.DataFrame(forecast_data)
        forecast_dates = pd.to_datetime(df_forecast['date'])


        months_ru = {
            1: 'январь', 2: 'февраль', 3: 'март', 4: 'апрель', 5: 'май', 6: 'июнь',
            7: 'июль', 8: 'август', 9: 'сентябрь', 10: 'октябрь', 11: 'ноябрь', 12: 'декабрь'
        }

        start_month_num = int(forecast_dates.iloc[0].month)
        start_year = int(forecast_dates.iloc[0].year)
        month_name = months_ru[start_month_num]

        plt.clf()
        plt.figure(figsize=(12, 6), dpi=100)


        plt.bar(forecast_dates, df_forecast['precip'], color='royalblue', alpha=0.5,
                label='Прогноз осадков (мм)', width=0.6)

        plt.plot(forecast_dates, df_forecast['temp_max'], color='gray',
                 linestyle='--', alpha=0.7, label='Базовый тренд T_max (°C)')

        norm_days = df_forecast[df_forecast['is_anomaly'] == 0]
        anomaly_days = df_forecast[df_forecast['is_anomaly'] == 1]

        if not norm_days.empty:
            plt.scatter(pd.to_datetime(norm_days['date']), norm_days['temp_max'],
                        color='seagreen', s=50, zorder=3, label='Норма климата')

        if not anomaly_days.empty:
            plt.scatter(pd.to_datetime(anomaly_days['date']), anomaly_days['temp_max'],
                        color='crimson', s=120, edgecolor='black', linewidth=1.5, marker='X',
                        zorder=4, label='КРИТИЧЕСКИЙ РИСК (Жара/Засуха)')


            for idx, row in anomaly_days.iterrows():
                plt.annotate(f"Риск! ({row['probability']:.1%})",
                             (pd.to_datetime(row['date']), row['temp_max']),
                             textcoords="offset points", xytext=(0, 10), ha='center',
                             fontsize=9, color='darkred', fontweight='bold')


        ax = plt.gca()
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d.%m'))


        if daysParam <= 15:
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
        elif daysParam <= 35:
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
        else:
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=3))

        plt.xticks(rotation=45, ha='right')

        plt.title(f'Прогноз метеорологических показателей с идентификацией рисков на {month_name} {start_year} года',
                  fontsize=12, fontweight='bold', pad=15)

        plt.xlabel('Календарная дата (День.Месяц)', fontsize=11)
        plt.ylabel('Значения показателей', fontsize=11)
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.legend(loc='lower right', frameon=True, shadow=True)

        plt.tight_layout()


        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(  'all')

        return templates.TemplateResponse(request,"predict.html",context={"forecast_data": forecast_data,"chart_img": img_base64,"horizon_days": daysParam,
                                                                          "has_anomalies": any(item["is_anomaly"] == 1 for item in forecast_data)})
    except Exception as e:
        plt.close('all')
        return HTMLResponse(content=f"Ошибка выполнения локального инференса:{str(e)}",status_code=500)

# uvicorn main:app --reload
