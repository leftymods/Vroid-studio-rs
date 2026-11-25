# vroid-rs

Собственный рендерер и редактор VRoid-аватаров на Rust: `wgpu` + MToon-шейдинг,
собственный загрузчик glTF/VRM, процедурные волосы, параметрические тело и лицо
как в VRoid Studio и экспорт VRM 1.0.

## Сборка

```bash
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.json   # headless lavapipe
cargo build --release
```

Бинарник: `target/release/vroid-rs`.

## Базовый аватар

`assets/base_female.glb` **не хранится в git** — он генерируется из установленного
VRoid Studio скриптом `tools/extract_avatar.py` (UnityPy, бандл
`VRoidStudio_Data/data.unity3d`):

```bash
/root/.venv-unity/bin/python tools/extract_avatar.py --hair --out /root/vroid/base_female.glb
```

Скрипт выбирает Body/Face/Hair из одного инстанса prefab, берёт bind-позу и
экспортирует UV, морфы лица и текстуры. Дальше всё редактирование — в Rust.

## Команды

```bash
vroid-rs info   MODEL.glb
vroid-rs render MODEL.glb --out shot.png --yaw 16 --pitch 4 --size 640x800
vroid-rs sheet  MODEL.glb --out turntable.png --views 8
vroid-rs expr   MODEL.glb --out faces.png
vroid-rs serve  MODEL.glb --port 8099
vroid-rs export MODEL.glb --out avatar.vrm --name "My Avatar" --author me
```

`serve` — интерактивный веб-редактор (встроенный `editor.html`): ракурс, цвета
кожи/глаз/волос, выражения, сплайны тела/лица, фон, скачивание PNG.
`GET /render?...` отдаёт PNG.

### Параметры тела и лица

Слайдеры принимают значения `[-1, 1]` (0 — базовая модель), работают в `render`
и `export`, а также в веб-редакторе. Тело (`--height`, `--head`, `--neck`,
`--arms`, `--legs`, `--shoulders`, `--hips`, `--waist`, `--bust`, `--thickness`)
редактирует трансформы костей и пере-скиннивает меши на CPU. Лицо (`--eye-size`,
`--eye-width`, `--eye-height`, `--eye-spacing`, `--eye-angle`, `--eye-y`,
`--brow-size`, `--brow-y`, `--brow-angle`, `--nose-size`, `--nose-y`,
`--mouth-size`, `--mouth-width`, `--mouth-y`) деформирует вершины лица:
области глаз/бровей/рта находятся по именам материалов, нос — по центральной
зоне кожи; все слои лица получают одинаковое смещение, чтобы декели не отклеивались.

### Экспорт VRM 1.0

`export` пишет GLB с расширениями `VRMC_vrm` (метаданные, humanoid-кости,
выражения) и `VRMC_materials_mtoon`. По умолчанию параметры тела/лица и цвета
запекаются в вершины/материалы, а expression-морфы остаются морф-таргетами VRM.
Флаг `--bake-expr` запекает выражения и удаляет морф-таргеты.
Метаданные: `--name`, `--version`, `--author` (повторяемо).

## Архитектура

| crate | назначение |
|---|---|
| `vroid-model` | загрузка glTF/VRM, материалы, MToon, скины |
| `vroid-avatar` | `AvatarDoc` (выражения, цвета), процедурные волосы |
| `vroid-render` | wgpu-рендерер: opaque/blend, toon + outline |
| `vroid-app` | CLI (`info/render/sheet/expr/serve/export`) и HTTP-сервер редактора |
| `vroid-model::export` | запись GLB/VRM 1.0 (`VRMC_vrm`, MToon) |

## Статус

- [x] bind-поза, корректный выбор инстанса аватара
- [x] face-морфы (41) и пресеты выражений
- [x] перекраска кожи/глаз/волос, MToon-шейдинг
- [x] процедурная причёска (боб с чёлкой) на Rust
- [x] веб-редактор
- [x] скелет + skin weights в GLB, CPU-скининг
- [x] параметры тела (рост, голова, плечи, руки, ноги, талия, грудь) через кости
- [x] параметры лица (глаза, брови, нос, рот)
- [x] экспорт VRM 1.0

## Лицензия

Код — MIT. Ассеты VRoid Studio не входят в репозиторий и используются только
локально.
