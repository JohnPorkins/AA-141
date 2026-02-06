#!/bin/bash

# 1. Создаем папку для сборки
DIR="ROBOT_EXPORT"
rm -rf $DIR
mkdir -p $DIR/models_backup

echo ">>> [1/5] Копирование скриптов..."
cp *.py $DIR/
cp requirements.txt $DIR/
# Копируем базу, если она есть
if [ -f "robot_memory.json" ]; then
    cp robot_memory.json $DIR/
fi

echo ">>> [2/5] Копирование модели SpeechBrain..."
if [ -d "tmp_model" ]; then
    cp -r tmp_model $DIR/models_backup/
else
    echo "⚠️ Папка tmp_model не найдена (модель скачается заново)"
fi

echo ">>> [3/5] Копирование модели InsightFace (скрытая папка)..."
# InsightFace хранит модели в домашней директории пользователя
if [ -d "$HOME/.insightface" ]; then
    cp -r $HOME/.insightface $DIR/models_backup/
else
    echo "⚠️ InsightFace кэш не найден"
fi

echo ">>> [4/5] Копирование кэша Torch (VAD)..."
# Silero VAD хранится в кэше torch
if [ -d "$HOME/.cache/torch" ]; then
    mkdir -p $DIR/models_backup/torch_cache
    cp -r $HOME/.cache/torch $DIR/models_backup/torch_cache/
fi

echo ">>> [5/5] Архивирование..."
tar -czvf robot_full_backup.tar.gz $DIR

echo "✅ ГОТОВО! Файл 'robot_full_backup.tar.gz' содержит всё."
echo "   Вес архива:"
du -h robot_full_backup.tar.gz
