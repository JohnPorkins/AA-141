import os
import sys
import json
import sqlite3
import math
from array import array
from datetime import datetime
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv
from pydantic import BaseModel
import speech_recognition as sr

def init_embeddings_db(folder: Path) -> str:
    """
    Инициализирует папку и SQLite DB для хранения фактов и embeddings.
    Возвращает путь к файлу базы данных (строка).
    """
    try:
        folder.mkdir(parents=True, exist_ok=True)
        db_path = folder / "embeddings.db"
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Создаём таблицу facts (если ещё не создана)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS facts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                dialogue_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                date TEXT NOT NULL,
                fact_text TEXT NOT NULL,
                people TEXT NOT NULL,
                objects TEXT NOT NULL,
                importance REAL DEFAULT 0.0,
                embedding TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Попытка добавить колонку importance (на случай старых схем)
        try:
            cursor.execute('ALTER TABLE facts ADD COLUMN importance REAL DEFAULT 0.0')
        except sqlite3.OperationalError:
            # колонка уже существует — игнорируем
            pass

        conn.commit()
        conn.close()
        return str(db_path)
    except Exception as e:
        print(f"Ошибка инициализации базы embeddings: {e}")
        # Вернём путь даже при ошибке (можно будет увидеть), но в случае ошибки многие операции упадут позже
        return str(folder / "embeddings.db")

# Initialize the Recognizer
r = sr.Recognizer()

# Use the microphone as source
with sr.Microphone() as source:
    print("Speak now...")
    # Adjust for ambient noise levels
    r.adjust_for_ambient_noise(source) 
    # Listen to the audio data
    audio_data = r.listen(source)

try:
    # Convert speech to text using Google Web Speech API
    text = r.recognize_google(audio_data)
    print(f"You said: {text}")
except sr.UnknownValueError:
    print("Google Speech Recognition could not understand audio")
except sr.RequestError as e:
    print(f"Could not request results from Google Speech Recognition service; {e}")
    # Створюємо таблицю для фактів (одна таблиця)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS facts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            dialogue_id TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            date TEXT NOT NULL,
            fact_text TEXT NOT NULL,
            people TEXT NOT NULL,   -- JSON-масив імен людей, пов'язаних з фактом
            objects TEXT NOT NULL,  -- JSON-масив об'єктів, пов'язаних з фактом
            importance REAL DEFAULT 0.0,  -- Важливість факта від 0.0 до 1.0
            embedding TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Додаємо колонку importance до існуючих таблиць, якщо її немає
    try:
        cursor.execute('ALTER TABLE facts ADD COLUMN importance REAL DEFAULT 0.0')
    except sqlite3.OperationalError:
        # Колонка вже існує, ігноруємо помилку
        pass
    
    conn.commit()
    conn.close()

import unicodedata

def generate_embedding(client, text):
    """Генерує embedding для тексту за допомогою OpenAI.

    Робимо нормалізацію Unicode та повторну спробу з безпечним UTF-8 кодуванням при помилках.
    Додаємо докладну діагностику, щоб відловити помилки кодування (ascii/UnicodeEncodeError).
    """
    import traceback
    try:
        # Гарантуємо, що передаємо звичайний str та нормалізуємо юнікод
        s = text if isinstance(text, str) else str(text)
        s = unicodedata.normalize('NFC', s)

        # Коротка перевірка — чи можна закодувати у utf-8
        try:
            s.encode('utf-8')
        except Exception as enc_err:
            print(f"generate_embedding: utf-8 encode check failed: {repr(enc_err)}; will replace invalid chars.")
            s = s.encode('utf-8', errors='replace').decode('utf-8')

        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=s
        )
        return response.data[0].embedding

    except UnicodeEncodeError as e:
        print("UnicodeEncodeError in generate_embedding:", repr(e))
        print("Input repr:", repr(s[:200]))
        print(traceback.format_exc())
        try:
            safe = s.encode('utf-8', errors='replace').decode('utf-8')
            response = client.embeddings.create(
                model="text-embedding-3-small",
                input=safe
            )
            return response.data[0].embedding
        except Exception as e2:
            print(f"Помилка при генерації embedding (повторна спроба): {repr(e2)}")
            print(traceback.format_exc())
            return None

    except Exception as e:
        # Логуємо деталі помилки для діагностики
        print(f"Exception in generate_embedding: {repr(e)}")
        print(traceback.format_exc())

        # Якщо повідомлення містить 'ascii' або помилка пов'язана з кодуванням — пробуємо fallback
        msg = str(e).lower()
        if 'ascii' in msg or isinstance(e, UnicodeEncodeError):
            try:
                safe = s.encode('utf-8', errors='replace').decode('utf-8')
                response = client.embeddings.create(
                    model="text-embedding-3-small",
                    input=safe
                )
                return response.data[0].embedding
            except Exception as e2:
                print(f"Помилка при генерації embedding (fallback не вдалася): {repr(e2)}")
                print(traceback.format_exc())
                return None

        return None

def save_fact_embedding(db_path, dialogue_id, timestamp, date, fact_text, people, objects, importance, embedding):
    """Зберігає факт та його embedding у SQLite базу даних та (якщо можливо) у sqlite-vec таблицю"""
    if embedding is None:
        return False
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Конвертуємо embedding (список float) у JSON рядок
        embedding_json = json.dumps(embedding)
        people_json = json.dumps(people, ensure_ascii=False)
        objects_json = json.dumps(objects, ensure_ascii=False)

        cursor.execute('''
            INSERT INTO facts 
            (dialogue_id, timestamp, date, fact_text, people, objects, importance, embedding)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (dialogue_id, timestamp, date, fact_text, people_json, objects_json, importance, embedding_json))

        fact_id = cursor.lastrowid

        # Спроба проініціалізувати sqlite-vec та додати запис у векторну таблицю
        if init_sqlite_vec(conn):
            try:
                # Пакуємо embedding у BLOB з float32 (вимагається sqlite-vec)
                vec_blob = sqlite3.Binary(array('f', embedding).tobytes())
                cursor.execute(
                    "INSERT OR REPLACE INTO facts_vec(rowid, embedding) VALUES (?, ?)",
                    (fact_id, vec_blob),
                )
            except Exception as e:
                # Якщо не вдалося оновити векторну таблицю — лог, але не падаємо
                print(f"Попередження: не вдалося оновити facts_vec: {e}")

        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"Помилка при збереженні embedding для факту: {e}")
        return False

def cosine_similarity(vec1, vec2):
    """Обчислює косинусну схожість між двома векторами"""
    if not vec1 or not vec2 or len(vec1) != len(vec2):
        return 0.0
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a * a for a in vec1))
    norm2 = math.sqrt(sum(b * b for b in vec2))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot_product / (norm1 * norm2)

def init_sqlite_vec(conn, dim: int = 1536) -> bool:
    """
    Ініціалізує розширення sqlite-vec та віртуальну таблицю facts_vec.
    Повертає True, якщо розширення успішно ініціалізоване, інакше False.
    """
    try:
        conn.enable_load_extension(True)

        # Пробуємо декілька типових назв розширення
        loaded = False
        for ext_name in ("sqlite-vec", "vec0"):
            try:
                conn.load_extension(ext_name)
                loaded = True
                break
            except Exception:
                continue

        if not loaded:
            # Якщо розширення не вдалося завантажити — просто повертаємо False,
            # далі код може використати повільний Python-пошук
            return False

        # Створюємо віртуальну таблицю для векторного пошуку (якщо її ще немає)
        conn.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS facts_vec USING vec0(embedding float[{dim}])"
        )
        return True
    except Exception as e:
        print(f"Попередження: не вдалося ініціалізувати sqlite-vec: {e}")
        return False

def search_similar_facts(client, db_path, query_text, top_n: int = 5, importance: float = 0.0):
    """
    По заданій фразі рахує embedding та шукає N найближчих фактів з бази.

    Використовує sqlite-vec, якщо можливо, інакше — резервний пошук у Python.
    Фільтрує факти за важливістю.

    Args:
        client: OpenAI клієнт
        db_path: Шлях до бази даних
        query_text: Текст для пошуку
        top_n: Кількість найближчих фактів для повернення
        importance: Поріг важливості факту (0.0-1.0), за замовчуванням 0.0.
                    - 0.0: включає тільки безглузді факти (importance = 0.0)
                    - 0.1-0.9: включає загальні знання та вище (importance >= importance)
                    - 1.0: включає тільки особисті знання (importance = 1.0)
                    
                    Правила оцінки importance:
                    - 0.0: безглузді факти ("неправда", "правда", "добре", "так", "ні")
                    - 0.1-0.9: загальні знання ("Яблука ростуть на деревах", "Небо синє")
                    - 1.0: особисті знання про людей ("Максим любить математику", "У Петра все добре")

    Повертає список словників:
    [
      {
        "id": int,
        "dialogue_id": str,
        "timestamp": str,
        "date": str,
        "fact_text": str,
        "people": list[str],
        "objects": list[str],
        "importance": float,
        "similarity": float
      },
      ...
    ]
    """
    # Генеруємо embedding для запиту
    query_embedding = generate_embedding(client, query_text)
    if query_embedding is None:
        return []

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
    except Exception as e:
        print(f"Помилка при відкритті бази embeddings: {e}")
        return []

    # Спершу намагаємось використати sqlite-vec
    if init_sqlite_vec(conn):
        try:
            # Пакуємо вектор запиту у BLOB float32
            query_blob = sqlite3.Binary(array('f', query_embedding).tobytes())
            cursor.execute(
                f"""
                SELECT f.id, f.dialogue_id, f.timestamp, f.date,
                       f.fact_text, f.people, f.objects, f.importance,
                       v.distance
                FROM facts_vec AS v
                JOIN facts AS f ON f.id = v.rowid
                WHERE v.embedding MATCH ? AND (f.importance IS NOT NULL AND f.importance >= ?)
                ORDER BY v.distance ASC
                LIMIT ?
                """,
                (query_blob, importance, top_n),
            )
            rows = cursor.fetchall()
            conn.close()

            results = []
            for row in rows:
                fact_id, dialogue_id, timestamp, date_str, fact_text, people_json, objects_json, fact_importance, distance = row
                
                # Фільтруємо за важливістю
                importance_value = fact_importance if fact_importance is not None else 0.0
                if importance == 0.0:
                    # Якщо importance = 0.0, включаємо тільки безглузді факти
                    if importance_value != 0.0:
                        continue
                else:
                    # Якщо importance > 0.0, включаємо факти з importance >= importance
                    if importance_value < importance:
                        continue
                
                try:
                    people = json.loads(people_json)
                except Exception:
                    people = []
                try:
                    objects = json.loads(objects_json)
                except Exception:
                    objects = []

                # Перетворюємо distance у "similarity": чим менша відстань, тим більша схожість
                similarity = 1.0 / (1.0 + float(distance))

                results.append({
                    "id": fact_id,
                    "dialogue_id": dialogue_id,
                    "timestamp": timestamp,
                    "date": date_str,
                    "fact_text": fact_text,
                    "people": people,
                    "objects": objects,
                    "importance": importance_value,
                    "similarity": similarity,
                })

            return results
        except Exception as e:
            print(f"Помилка при векторному пошуку через sqlite-vec, використовую резервний пошук: {e}")
            # Якщо щось пішло не так — падаємо у резервний варіант нижче

    # Резервний повільний варіант: читаємо всі факти та рахуємо косинусну схожість у Python
    try:
        cursor.execute(
            """
            SELECT id, dialogue_id, timestamp, date, fact_text, people, objects, importance, embedding
            FROM facts
            """
        )
        rows = cursor.fetchall()
        conn.close()
    except Exception as e:
        print(f"Помилка при читанні з бази embeddings: {e}")
        return []

    results = []
    for row in rows:
        fact_id, dialogue_id, timestamp, date_str, fact_text, people_json, objects_json, fact_importance, embedding_json = row
        try:
            fact_embedding = json.loads(embedding_json)
            similarity = cosine_similarity(query_embedding, fact_embedding)
        except Exception:
            # Якщо щось не так з конкретним записом — пропускаємо
            continue

        # Фільтруємо за важливістю
        importance_value = fact_importance if fact_importance is not None else 0.0
        if importance == 0.0:
            # Якщо importance = 0.0, включаємо тільки безглузді факти
            if importance_value != 0.0:
                continue
        else:
            # Якщо importance > 0.0, включаємо факти з importance >= importance
            if importance_value < importance:
                continue

        try:
            people = json.loads(people_json)
        except Exception:
            people = []

        try:
            objects = json.loads(objects_json)
        except Exception:
            objects = []

        results.append({
            "id": fact_id,
            "dialogue_id": dialogue_id,
            "timestamp": timestamp,
            "date": date_str,
            "fact_text": fact_text,
            "people": people,
            "objects": objects,
            "importance": importance_value,
            "similarity": similarity,
        })

    # Сортуємо за схожістю (від більшої до меншої) і повертаємо top_n
    results.sort(key=lambda x: x["similarity"], reverse=True)
    return results[:top_n]

class FactItem(BaseModel):
    fact: str
    people: list[str]
    objects: list[str]
    importance: float = 0.0


class DialogueExtract(BaseModel):
    facts: list[FactItem]
    date: str


def create_extract(client, messages, dialogue_date):
    """Створює вижимку діалогу з фактами, де для кожного факта вказані люди та об'єкти"""
    # Формуємо текст діалогу для аналізу ТІЛЬКИ з повідомлень користувача
    dialogue_text = "\n".join([
        f"{msg['role']}: {msg['content']}"
        for msg in messages
    ])
    
    # Промпт для аналізу діалогу
    analysis_prompt = f"""Проаналізуй наступний діалог та створи структуровану вижимку.

Діалог:
{dialogue_text}

Створи об'єкт з такими полями, використовуючи ТІЛЬКИ інформацію з висловлювань користувача (USER:):
- "facts": масив об'єктів, де КОЖЕН об'єкт описує ОКРЕМИЙ факт.
  Для кожного факта повинні бути поля:
  - "fact": текст факта одним реченням
  - "people": масив імен людей, які стосуються саме цього факта (якщо немає — порожній масив [])
  - "objects": масив об'єктів, пов'язаних з фактом (якщо немає — порожній масив [])
  - "importance": важливість факту (float від 0.0 до 1.0), ОБОВ'ЯЗКОВО визначити за такими правилами:
     * 0.0 — безглузді факти без конкретного змісту (наприклад: "неправда", "правда", "добре", "так", "ні", "ок", "гаразд")
     * 0.1-0.9 — загальні знання та факти про світ (наприклад: "Яблука ростуть на деревах", "Небо синє", "Вода кипить при 100 градусах")
     * 1.0 — особисті знання про конкретних людей, їх уподобання, події з їх життя (наприклад: "Максим любить математику", "У Петра все добре", "Анна працює в IT")
- "date": дата діалогу у форматі YYYY-MM-DD

Відповідай без додаткового тексту. Якщо якогось елемента немає, використовуй порожній масив [].

Приклад формату:
{{
  "facts": [
    {{
      "fact": "Максим любить математику",
      "people": ["Максим"],
      "objects": ["математика"],
      "importance": 1.0
    }},
    {{
      "fact": "У Петра все добре",
      "people": ["Петро"],
      "objects": [],
      "importance": 1.0
    }},
    {{
      "fact": "Яблука ростуть на деревах",
      "people": [],
      "objects": ["яблука", "дерева"],
      "importance": 0.5
    }},
    {{
      "fact": "Небо синє",
      "people": [],
      "objects": ["небо"],
      "importance": 0.3
    }},
    {{
      "fact": "добре",
      "people": [],
      "objects": [],
      "importance": 0.0
    }},
    {{
      "fact": "неправда",
      "people": [],
      "objects": [],
      "importance": 0.0
    }}
  ],
  "date": "2024-01-15"
}}"""

    try:
        response = client.responses.parse(
            model="gpt-4o-mini",
            input=[
                {
                    "role": "system",
                    "content": (
                        "Ти експерт з аналізу текстів. "
                        "Проаналізуй висловлювання користувача та заповни Pydantic-схему "
                        "з фактами, людьми та об'єктами. "
                        "\n\nКРИТИЧНО ВАЖЛИВО правильно визначити importance для кожного факту:\n"
                        "- 0.0 — безглузді факти без конкретного змісту (однослівні відповіді типу 'так', 'ні', 'добре', 'неправда', 'правда')\n"
                        "- 0.1-0.9 — загальні знання про світ, природу, науку, культуру (факти, які стосуються всіх, а не конкретної особи)\n"
                        "- 1.0 — особисті знання про конкретних людей: їх уподобання, події з їх життя, особисті характеристики\n"
                        "\nЯкщо чогось немає, використовуй порожній список."
                    ),
                },
                {"role": "user", "content": analysis_prompt},
            ],
            text_format=DialogueExtract,
            temperature=0.3,
        )

        extract_model: DialogueExtract = response.output_parsed
        extract_data = extract_model.model_dump()

        # Перезаписуємо дату діалогу на фактичну
        extract_data["date"] = dialogue_date

        return extract_data
    except Exception as e:
        print(f"\nПомилка при створенні вижимки: {e}")
        # Повертаємо базову структуру з датою
        return {
            "facts": [],
            "date": dialogue_date
        }

def save_dialogue(messages, dialogues_dir, summaries_dir, client, db_path):
    """Зберігає діалог у файл та створює вижимку з embeddings для фактів (з людьми та об'єктами для кожного факта)"""
    if not messages:
        return
    
    # Створюємо папку, якщо її немає
    dialogues_dir.mkdir(exist_ok=True)
    summaries_dir.mkdir(exist_ok=True)
    
    # Створюємо ім'я файлу з датою та часом
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    dialogue_date = datetime.now().strftime("%Y-%m-%d")
    dialogue_id = f"dialogue_{timestamp}"
    filename = dialogues_dir / f"{dialogue_id}.json"
    
    # Створюємо структуру для збереження
    dialogue_data = {
        "timestamp": datetime.now().isoformat(),
        "messages": messages
    }
    
    # Зберігаємо у JSON форматі
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(dialogue_data, f, ensure_ascii=False, indent=2)
    
    print(f"\nДіалог збережено: {filename}")
    
    # Створюємо вижимку
    print("Створюю вижимку діалогу...")
    extract_data = create_extract(client, messages, dialogue_date)
    
    # Зберігаємо вижимку
    extract_filename = summaries_dir / f"extract_{timestamp}.json"
    with open(extract_filename, 'w', encoding='utf-8') as f:
        json.dump(extract_data, f, ensure_ascii=False, indent=2)
    
    print(f"Вижимка збережена: {extract_filename}")
    
    # Генеруємо та зберігаємо embeddings для окремих фактів
    print("Генерую embeddings для фактів...")
    for fact_item in extract_data.get("facts", []):
        # Очікуємо структуру:
        # {
        #   "fact": "...",
        #   "people": [...],
        #   "objects": [...],
        #   "importance": 0.0-1.0
        # }
        fact_text = fact_item.get("fact")
        if not fact_text:
            continue

        people = fact_item.get("people", [])
        objects = fact_item.get("objects", [])
        importance = fact_item.get("importance", 0.0)

        # Формуємо текст для embedding: факт + пов'язані люди та об'єкти
        combined_parts = [fact_text]
        if people:
            combined_parts.append("Люди: " + ", ".join(people))
        if objects:
            combined_parts.append("Об'єкти: " + ", ".join(objects))
        embedding_text = "\n".join(combined_parts)

        fact_embedding = generate_embedding(client, embedding_text)
        if fact_embedding:
            saved = save_fact_embedding(
                db_path=db_path,
                dialogue_id=dialogue_id,
                timestamp=timestamp,
                date=dialogue_date,
                fact_text=fact_text,
                people=people,
                objects=objects,
                importance=importance,
                embedding=fact_embedding,
            )
            if not saved:
                print(f"Не вдалося зберегти embedding для факта: {fact_text}")

def listen_from_mic(timeout: int = 5, phrase_time_limit: int = 30, lang: str = "uk-UA") -> str:
    """
    Слушает микрофон и возвращает распознанный текст.
    Если распознание не удалось или микрофон недоступен — возвращает пустую строку.
    """
    r = sr.Recognizer()
    try:
        with sr.Microphone() as mic:
            r.adjust_for_ambient_noise(mic, duration=1)
            print("Слушаю (скажите сообщение)...")
            audio = r.listen(mic, timeout=timeout, phrase_time_limit=phrase_time_limit)
    except Exception as e:
        print(f"Микрофон недоступен або помилка захоплення аудіо: {e}")
        return ""

    try:
        text = r.recognize_google(audio, language=lang)
        print(f"Розпізнано: {text}")
        return text
    except sr.UnknownValueError:
        print("Не вдалося розпізнати мову.")
        return ""
    except sr.RequestError as e:
        print(f"Помилка сервісу розпізнавання: {e}")
        return ""

def main():
    # Robust loading of OPENAI_API_KEY: check env first, then search for .env in several places
    def try_load_openai_key():
        # 1) already present as environment variable
        key = os.getenv("OPENAI_API_KEY")
        if key:
            return True, "environment"

        # 2) use find_dotenv() to search upwards from current working directory
        try:
            dotenv_path = find_dotenv()
        except Exception:
            dotenv_path = ""

        if dotenv_path:
            load_dotenv(dotenv_path)
            if os.getenv("OPENAI_API_KEY"):
                return True, f"find_dotenv:{dotenv_path}"

        # 3) search from the script directory upward
        script_dir = Path(__file__).resolve().parent
        for pdir in [script_dir] + list(script_dir.parents)[:6]:
            p = pdir / ".env"
            if p.exists():
                load_dotenv(str(p))
                if os.getenv("OPENAI_API_KEY"):
                    return True, str(p)

        # 4) check current working directory
        p = Path.cwd() / ".env"
        if p.exists():
            load_dotenv(str(p))
            if os.getenv("OPENAI_API_KEY"):
                return True, str(p)

        # 5) try to parse any found .env files manually (fallback)
        tried = [str(Path.cwd() / ".env"), str(script_dir / ".env")]
        for t in tried:
            try:
                tp = Path(t)
                if tp.exists():
                    content = tp.read_text(encoding="utf-8")
                    for line in content.splitlines():
                        if line.strip().startswith("OPENAI_API_KEY"):
                            parts = line.split("=", 1)
                            if len(parts) == 2:
                                k = parts[1].strip()
                                if k:
                                    os.environ["OPENAI_API_KEY"] = k
                                    return True, f"parsed:{t}"
            except Exception:
                continue

        return False, None

    ok, src = try_load_openai_key()
    if not ok:
        print("Помилка: Не знайдено OPENAI_API_KEY.")
        print("Перевірте, що ключ вказаний у файлі `.env` у корені проекту або як системна змінна середовища.")
        print("Приклад у файлі .env:")
        print("OPENAI_API_KEY=your_api_key_here")
        print("Debug: Current working directory:", os.getcwd())
        print("Debug: Script path:", Path(__file__).resolve())
        sys.exit(1)
    else:
        api_key = os.getenv("OPENAI_API_KEY")
        # sanitize quotes if present
        if api_key and ((api_key.startswith('"') and api_key.endswith('"')) or (api_key.startswith("'") and api_key.endswith("'"))):
            api_key = api_key[1:-1]
            os.environ["OPENAI_API_KEY"] = api_key
        masked = api_key[:4] + "..." + api_key[-4:] if api_key and len(api_key) > 8 else "****"
        print(f"OPENAI_API_KEY loaded from {src} (masked): {masked}")
    
    # Створюємо папки для діалогів та вижимок
    dialogues_dir = Path("dialogues")
    summaries_dir = Path("summaries")
    
    # Створюємо окрему папку для бази даних embeddings
    embeddings_db_folder = Path("embeddings_db")
    
    # Ініціалізуємо базу даних для embeddings
    db_path = init_embeddings_db(embeddings_db_folder)
    
    # Ініціалізація клієнта OpenAI
    client = OpenAI(api_key=api_key)
    
    print("=" * 60)
    print("Діалог з OpenAI")
    print("Введіть 'exit', 'quit' або 'вихід' для завершення")
    print("=" * 60)
    print()
    
    # Історія розмови
    messages = []
    
    while True:
        try:
            # Получение ввода от пользователя через микрофон
            user_input = listen_from_mic().strip()

            if not user_input:
                continue

            # Шукаємо найближчі факти до поточного запиту
            similar_facts = search_similar_facts(client, str(db_path), user_input, top_n=5, importance=0.2)

            # Формуємо текстовий контекст з релевантних фактів
            retrieved_context_parts = []
            for i, fact in enumerate(similar_facts, start=1):
                retrieved_context_parts.append(
                    f"- Факт {i} (дата: {fact['date']}): {fact['fact_text']}"
                )
                if fact.get("people"):
                    retrieved_context_parts.append(
                        "  Люди: " + ", ".join(fact["people"])
                    )
                if fact.get("objects"):
                    retrieved_context_parts.append(
                        "  Об'єкти: " + ", ".join(fact["objects"])
                    )
            retrieved_context = "\n".join(retrieved_context_parts) if retrieved_context_parts else "Немає збережених релевантних фактів."

            # Системне повідомлення з піднятим контекстом із бази
            context_message = {
                "role": "system",
                "content": (
                    "Нижче наведені релевантні збережені факти з попередніх діалогів. "
                    "Використовуй їх як додатковий контекст, але якщо вони не підходять, можеш їх ігнорувати.\n\n"
                    f"{retrieved_context}"
                )
            }

            # Формуємо список повідомлень для запиту: контекст + вся історія + поточний запит
            request_messages = [context_message] + messages + [
                {"role": "user", "content": user_input}
            ]

            # Додаємо поточне повідомлення користувача в історію
            messages.append({
                "role": "user",
                "content": user_input
            })

            # Відправка запиту до OpenAI
            print("\nAI: ", end="", flush=True)
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=request_messages,
                stream=True
            )
            
            # Обробка потокової відповіді
            assistant_message = ""
            for chunk in response:
                if chunk.choices[0].delta.content is not None:
                    content = chunk.choices[0].delta.content
                    print(content, end="", flush=True)
                    assistant_message += content
            
            print("\n")
            
            # Додавання відповіді асистента до історії
            messages.append({
                "role": "assistant",
                "content": assistant_message
            })
            
            print()
            
        except KeyboardInterrupt:
            # Зберігаємо діалог при перериванні
            if messages:
                save_dialogue(messages, dialogues_dir, summaries_dir, client, db_path)
            print("\n\nПерервано користувачем. До побачення!")
            break
        except Exception as e:
            print(f"\nПомилка: {e}")
            print("Спробуйте ще раз або введіть 'exit' для виходу.\n")

if __name__ == "__main__":
    main()

