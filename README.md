# vroid-rs

Собственный рендерер и редактор VRoid-аватаров на Rust: `wgpu` + MToon-шейдинг,
собственный загрузчик glTF/VRM, процедурные волосы и (в работе) параметрическое
тело как в VRoid Studio.

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
```

`serve` — интерактивный веб-редактор (встроенный `editor.html`): ракурс, цвета
кожи/глаз/волос, выражения, фон, скачивание PNG. `GET /render?...` отдаёт PNG.

## Архитектура

| crate | назначение |
|---|---|
| `vroid-model` | загрузка glTF/VRM, материалы, MToon, скины |
| `vroid-avatar` | `AvatarDoc` (выражения, цвета), процедурные волосы |
| `vroid-render` | wgpu-рендерер: opaque/blend, toon + outline |
| `vroid-app` | CLI (`info/render/sheet/expr/serve`) и HTTP-сервер редактора |

## Статус

- [x] bind-поза, корректный выбор инстанса аватара
- [x] face-морфы (41) и пресеты выражений
- [x] перекраска кожи/глаз/волос, MToon-шейдинг
- [x] процедурная причёска (боб с чёлкой) на Rust
- [x] веб-редактор
- [ ] скелет + skin weights в GLB, CPU-скининг
- [ ] параметры тела (рост, голова, плечи, руки, ноги, талия, грудь) через кости
- [ ] параметры лица (глаза, брови, нос, рот)
- [ ] экспорт VRM 1.0

## Лицензия

Код — MIT. Ассеты VRoid Studio не входят в репозиторий и используются только
локально.
