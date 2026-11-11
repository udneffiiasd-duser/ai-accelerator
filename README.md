# ai-accelerator

Набор утилит для **ускорения инференса** нейросетей: динамический батчинг,
кэширование ответов и подсказки по квантизации.

## Возможности

- **Динамический батчинг** — объединение запросов в один вызов модели
- **Кэш ответов** — LRU с TTL, экономит вычисления
- **Профилировщик** — замер задержки и пропускной способности
- **Квантизация** — рекомендации по снижению точности без потери качества

## Установка

```bash
pip install -r requirements.txt
```

## Пример

```python
from accelerator import BatchEngine, ResponseCache

cache = ResponseCache(maxsize=1024, ttl=300)
engine = BatchEngine(max_batch=32, timeout_ms=10)

@engine.infer
def model(text: str) -> str:
    return text[::-1]

print(model("привет"))  # мгновенно, при повторе — из кэша
```

## Лицензия

MIT
