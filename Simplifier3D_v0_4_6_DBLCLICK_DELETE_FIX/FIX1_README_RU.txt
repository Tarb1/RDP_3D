Simplifier3D v0.1 CORRECT FIX1
================================

Причина ошибки предыдущего пакета:
geometry3d_module.py импортирует geometry3d_core.py, но этот файл по ошибке
не попал в Simplifier3D_v0_1_CORRECT.zip. Поэтому собранный EXE завершался:

    ModuleNotFoundError: No module named 'geometry3d_core'

Исправлено:
- geometry3d_core.py возвращен в проект;
- build_exe.bat проверяет наличие обоих 3D-модулей;
- перед сборкой запускается self_test_3d_core.py;
- перед сборкой проверяются импорты;
- PyInstaller получает явные --hidden-import geometry3d_core/geometry3d_module.

Сборка:
    build_exe.bat

После запуска должны быть вкладки:
    Axis Simplifier
    Geometria AB-BC
    Geometry 3D
