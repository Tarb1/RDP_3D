Simplifier3D v0.1 — правильная ветка от рабочего v26
====================================================

ВАЖНО
------
Запускается ТОЛЬКО один главный файл:
    rdp_simplifier26.py
или:
    RUN.bat

Для EXE:
    build_exe.bat

После запуска в верхней строке ОБЯЗАТЕЛЬНО должны быть 3 вкладки:
    1. Axis Simplifier
    2. Geometria AB–BC
    3. Geometry 3D

Заголовок окна:
    Simplifier3D v0.1 — based on RDP Simplifier 26

Если третьей вкладки нет, запущена не эта версия.

ЭТАП 1
------
Старые Axis Simplifier и Geometria AB–BC оставлены как в v26.
Добавлена только вкладка Geometry 3D.

На Geometry 3D:
- MODEL — 3D LINE -> после SIMPLIFY XYZ должно быть 2 точки.
- MODEL — 3 SEGMENTS -> после SIMPLIFY XYZ должно быть 4 точки.

Логика Target points такая же, как в v26:
- Target points = 0 -> оператор задаёт Max XYZ error.
- Target points > 0 -> программа автоматически ищет epsilon.

Дуги пока НЕ перенесены. Это следующий этап после проверки 3D RDP.
