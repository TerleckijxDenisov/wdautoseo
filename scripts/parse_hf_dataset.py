"""
Парсер датасета Wildberries из HuggingFace

Этот скрипт распаковывает сжатый файл .json.zst с товарами
и сохраняет нужные поля в CSV для дальнейшей обработки.

Использует потоковую обработку для работы с большими файлами.

Использование: python3 scripts/parse_hf_dataset.py --limit 1000000
"""
import zstandard as zstd
import json
import csv
import os
import io
import argparse


# ===========================================================================
# Основная функция парсинга
# ===========================================================================
def parse_zst_jsonl(input_file, output_file, limit=1000000, append=False):
    """
    Распаковывает .jsonl.zst файл и извлекает нужные поля.
    
    Параметры:
    - input_file: путь к сжатому файлу
    - output_file: путь к выходному CSV
    - limit: максимальное количество записей
    - append: добавлять к существующему файлу или перезаписать
    """
    print(f"Парсинг {input_file} -> {output_file} (Лимит: {limit} записей)...")
    
    # Создаем декомпрессор для формата Zstandard
    dctx = zstd.ZstdDecompressor()
    
    with open(input_file, 'rb') as f_in:
        # Открываем поток для чтения сжатых данных
        with dctx.stream_reader(f_in) as reader:
            # Оборачиваем бинарный поток в текстовый для построчного чтения
            text_stream = io.TextIOWrapper(reader, encoding='utf-8')
            
            mode = 'a' if append else 'w'
            with open(output_file, mode, newline='', encoding='utf-8') as f_out:
                # Определяем поля которые нам нужны
                fieldnames = [
                    'sku_id',       # Артикул товара
                    'category',     # Категория (например "Платья")  
                    'raw_title',    # Название товара
                    'description',  # Описание
                    'brand',        # Бренд
                    'price',        # Цена
                    'rating',       # Рейтинг
                    'feedbacks',    # Количество отзывов
                    'colors'        # Цвета
                ]
                
                writer = csv.DictWriter(f_out, fieldnames=fieldnames, delimiter='\t')
                
                # Записываем заголовок только если создаем новый файл
                if not append:
                    writer.writeheader()
                
                count = 0
                for line in text_stream:
                    try:
                        item = json.loads(line)
                        
                        # Функция для очистки текста от спецсимволов
                        def clean_text(text):
                            if not text:
                                return ""
                            return str(text).replace('\n', ' ').replace('\r', ' ').replace('\t', ' ').strip()
                        
                        # Записываем строку в CSV
                        # API Wildberries может возвращать поля с разными именами,
                        # поэтому проверяем несколько вариантов
                        writer.writerow({
                            'sku_id': item.get('nm_id') or item.get('id'),
                            'category': clean_text(item.get('subj_name') or item.get('subjectName')),
                            'raw_title': clean_text(item.get('imt_name') or item.get('name')),
                            'description': clean_text(item.get('description', '')),
                            'brand': clean_text(item.get('brand_name') or item.get('brand')),
                            'price': item.get('price_u', 0) / 100 if item.get('price_u') else 0,
                            'rating': item.get('rating', 0),
                            'feedbacks': item.get('feedbacks', 0),
                            'colors': clean_text(", ".join(item.get('colors', [])) if isinstance(item.get('colors'), list) else "")
                        })
                        
                        count += 1
                        
                        # Выводим прогресс каждые 100к записей
                        if count % 100000 == 0:
                            print(f"Обработано {count} записей...")
                        
                        # Останавливаемся при достижении лимита
                        if count >= limit:
                            break
                            
                    except Exception as e:
                        # Пропускаем битые строки
                        continue
                        
    print(f"Готово! Успешно распарсено {count} записей.")


# ===========================================================================
# Точка входа
# ===========================================================================
def main():
    parser = argparse.ArgumentParser(description="Парсер датасета wb-products с HuggingFace")
    parser.add_argument("--input", type=str, default="data/raw/basket-01.json.zst", 
                        help="Входной .zst файл")
    parser.add_argument("--output", type=str, default="data/raw/real_wb_data.csv", 
                        help="Выходной .csv файл")
    parser.add_argument("--limit", type=int, default=1000000, 
                        help="Максимум записей для парсинга")
    
    args = parser.parse_args()
    
    input_file = args.input
    output_file = args.output
    
    if os.path.exists(input_file):
        parse_zst_jsonl(input_file, output_file, args.limit)
    else:
        print(f"Ошибка: файл {input_file} не найден.")
        print("Скачайте датасет с HuggingFace: nyuuzyou/wb-products")


if __name__ == "__main__":
    main()
