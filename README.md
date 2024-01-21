# BostonHouseproject
## Джерело Датасету
Датасет взятий з [Kaggle](https://www.kaggle.com/datasets/schirmerchad/bostonhoustingmlnd)

## Огляд
Дані про житло в Бостоні були зібрані в 1978 році, і кожен із 506 записів представляє сукупні дані про 14 об’єктів для будинків із різних передмість у Бостоні, штат Массачусетс.

### Файли
- **housing.csv**
- *CRIM*: Рівень злочинності на душу населення за містами.
- *ZN*: Частка житлової забудови для ділянок площею понад 25 000 кв.м.
- *INDUS*: Частка площ під нероздрібну торгівлю за містом.
- *CHAS*: Річка Чарльз - фіктивна змінна (1, якщо урочище межує з річкою; 0 в іншому випадку).
- *NOX*: Концентрація оксидів азоту (частин на 10 мільйонів).
- *RM*: Середня кількість кімнат на одну оселю.
- *AGE*: Частка власників житла, збудованого до 1940 року.
- *DIS*: Середньозважена відстань до п'яти центрів зайнятості Бостона.
- *RAD*: Індекс доступності до радіальних магістралей.
- *TAX*: Ставка податку на повну вартість нерухомості на 10 000 доларів США.
- *PTRATIO*: Співвідношення кількості учнів та вчителів у місті.
- *B*: 1000 * (Bk - 0,63)^2, де Bk - частка темношкірого населення в місті.
- *LSTAT*: Відсоток населення з низьким соціальним статусом.
- *MEDV*: Медіана вартості приватного житла у $1000.

## Використання
Цей датасет був використаний нами для прогнозування MEDV: Медіана вартості приватного житла у $1000.
# Boston Housing Price Prediction App
Наша програма прогнозує медіану вартості приватного житла у $1000 в бостонні відносно введених користувачем даних, використовуючи при цьому нашу навчену модель, для навчання моделі ми використакли наш CustomLinearRegression та порахували метрики для нашої моделі а саме Model	R2 Score	Adjusted R2 Score	Cross Validated R2 Score	RMSE.
![App](https://github.com/ViktorPrystai/BostonHouseproject/blob/main/BostonHouseproject/screnshots/BostonHouse%20price%20prediction.jpg)
# Metrics
![App](https://github.com/ViktorPrystai/BostonHouseproject/blob/main/BostonHouseproject/screnshots/metrics.jpg)
