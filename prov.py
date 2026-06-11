import cv2
import json
import sys
import numpy as np
from openai import OpenAI
 
client = OpenAI(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio"
)
 
# ─────────────────────────────────────────
#  Обработчик изображений
# ─────────────────────────────────────────
 
class ImageProcessor:
    def __init__(self, image_path):
        self.image_path = image_path
        self.img = cv2.imread(image_path)
        if self.img is None:
            raise FileNotFoundError(f"Файл не найден: {image_path}")
        self.is_grayscale = False
 
    def rotate_image(self, angle: int):
        if angle in (90, 180, 270):
            codes = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}
            self.img = cv2.rotate(self.img, codes[angle])
        else:
            h, w = self.img.shape[:2]
            cx, cy = w // 2, h // 2
            M = cv2.getRotationMatrix2D((cx, cy), -angle, 1.0)
            cos, sin = abs(M[0, 0]), abs(M[0, 1])
            new_w = int(h * sin + w * cos)
            new_h = int(h * cos + w * sin)
            M[0, 2] += new_w / 2 - cx
            M[1, 2] += new_h / 2 - cy
            self.img = cv2.warpAffine(self.img, M, (new_w, new_h),
                                      borderMode=cv2.BORDER_CONSTANT,
                                      borderValue=(0, 0, 0))
        return f"Изображение повернуто на {angle}°"
 
    def resize_image(self, width: int, height: int):
        self.img = cv2.resize(self.img, (width, height))
        return f"Размер изменён на {width}×{height}"
 
    def flip_image(self, direction: str):
        self.img = cv2.flip(self.img, 1 if direction == "horizontal" else 0)
        return f"Изображение отражено {'горизонтально' if direction == 'horizontal' else 'вертикально'}"
 
    def blur_image(self, kernel_size: int):
        if kernel_size % 2 == 0:
            kernel_size += 1
        self.img = cv2.GaussianBlur(self.img, (kernel_size, kernel_size), 0)
        return f"Изображение размыто (ядро {kernel_size})"
 
    def convert_to_grayscale(self, mode: str = "standard"):
        if self.is_grayscale or len(self.img.shape) == 2:
            self.img = cv2.cvtColor(self.img, cv2.COLOR_GRAY2BGR)
            self.is_grayscale = False
 
        b, g, r = cv2.split(self.img.astype(np.float32))
 
        if mode == "standard":
            gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        elif mode == "hc":
            lab = cv2.cvtColor(self.img, cv2.COLOR_BGR2LAB)
            l, a, b_ch = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            lab_eq = cv2.merge([clahe.apply(l), a, b_ch])
            gray = cv2.cvtColor(cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR), cv2.COLOR_BGR2GRAY)
        elif mode == "soft":
            blurred = cv2.GaussianBlur(self.img, (3, 3), 0)
            gray = cv2.addWeighted(cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY),
                                   0.85, np.full_like(blurred[:, :, 0], 255), 0.15, 0)
        elif mode == "film":
            gray_f = 0.27 * r + 0.53 * g + 0.20 * b
            gray = np.clip(gray_f, 0, 255).astype(np.uint8)
            noise = np.random.normal(0, 4, gray.shape).astype(np.int16)
            gray = np.clip(gray.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        elif mode == "infrared":
            gray_f = 0.07 * r + 0.72 * g + 0.21 * b
            gray = cv2.equalizeHist(np.clip(gray_f, 0, 255).astype(np.uint8))
        else:
            gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
 
        self.img = gray
        self.is_grayscale = True
        return f"Чёрно-белый режим: {mode}"
 
    def save_result(self, output_path: str = "result.jpg"):
        cv2.imwrite(output_path, self.img)
        return f"Сохранено в {output_path}"
 
    def get_info(self):
        h, w = self.img.shape[:2]
        channels = 1 if len(self.img.shape) == 2 else self.img.shape[2]
        mode = "ч/б" if channels == 1 else "цветное"
        return f"Текущий размер: {w}×{h}, режим: {mode}"
 
 
# ─────────────────────────────────────────
#  Описание инструментов для LLM
# ─────────────────────────────────────────
 
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "rotate_image",
            "description": "Повернуть изображение на заданный угол.",
            "parameters": {
                "type": "object",
                "properties": {
                    "angle": {"type": "integer", "description": "Угол поворота в градусах (например 90, 180, 45)"}
                },
                "required": ["angle"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "resize_image",
            "description": "Изменить размер (разрешение) изображения.",
            "parameters": {
                "type": "object",
                "properties": {
                    "width":  {"type": "integer", "description": "Ширина в пикселях"},
                    "height": {"type": "integer", "description": "Высота в пикселях"}
                },
                "required": ["width", "height"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "flip_image",
            "description": "Отразить изображение зеркально.",
            "parameters": {
                "type": "object",
                "properties": {
                    "direction": {
                        "type": "string",
                        "enum": ["horizontal", "vertical"],
                        "description": "horizontal — влево/вправо, vertical — вверх/вниз"
                    }
                },
                "required": ["direction"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "blur_image",
            "description": "Размыть изображение. Чем больше kernel_size, тем сильнее размытие.",
            "parameters": {
                "type": "object",
                "properties": {
                    "kernel_size": {
                        "type": "integer",
                        "description": "Размер ядра размытия: 5 — слабое, 15 — среднее, 31 — сильное"
                    }
                },
                "required": ["kernel_size"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "convert_to_grayscale",
            "description": (
                "Перевести изображение в чёрно-белый режим. "
                "Режимы: standard — обычный, hc — высококонтрастный (CLAHE), "
                "soft — мягкий плёночный, film — имитация Kodak T-MAX, infrared — инфракрасный эффект."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "mode": {
                        "type": "string",
                        "enum": ["standard", "hc", "soft", "film", "infrared"],
                        "description": "Стиль чёрно-белого преобразования"
                    }
                },
                "required": ["mode"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "save_result",
            "description": "Сохранить результат в файл.",
            "parameters": {
                "type": "object",
                "properties": {
                    "output_path": {
                        "type": "string",
                        "description": "Путь для сохранения (по умолчанию result.jpg)"
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_info",
            "description": "Получить информацию о текущем состоянии изображения (размер, цветность).",
            "parameters": {"type": "object", "properties": {}}
        }
    },
]
 
SYSTEM_PROMPT = (
    "Ты — ассистент для редактирования изображений. "
    "Пользователь описывает на русском языке, что нужно сделать с картинкой. "
    "Твоя задача — вызвать нужные инструменты (можно несколько за раз) и кратко сообщить о результате. "
    "Не отвечай текстом, если нужно выполнить операцию — используй инструмент. "
    "Если запрос непонятен или не относится к редактированию — вежливо уточни."
)
 
# ─────────────────────────────────────────
#  Диалог с нейросетью
# ─────────────────────────────────────────
 
def process_command(user_input: str, processor: ImageProcessor, history: list) -> list:
    """Отправляет запрос в LLM, выполняет tool calls, возвращает обновлённую историю."""
    history.append({"role": "user", "content": user_input})
 
    while True:
        response = client.chat.completions.create(
            model="local-model",
            messages=[{"role": "system", "content": SYSTEM_PROMPT}] + history,
            tools=TOOLS,
            tool_choice="auto",
            temperature=0.0
        )
        message = response.choices[0].message
        history.append(message.model_dump(exclude_unset=False))
 
        # Если нет tool calls — просто выводим ответ
        if not message.tool_calls:
            if message.content:
                print(f"\n🤖 {message.content}")
            break
 
        # Выполняем все вызванные инструменты
        tool_results = []
        for tc in message.tool_calls:
            func_name = tc.function.name
            args = json.loads(tc.function.arguments)
            print(f"  ⚙️  Вызов: {func_name}({args})")
 
            if hasattr(processor, func_name):
                result = getattr(processor, func_name)(**args)
                print(f"  ✅ {result}")
            else:
                result = f"Функция {func_name} не найдена."
                print(f"  ❌ {result}")
 
            tool_results.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result
            })
 
        history.extend(tool_results)
        # Продолжаем цикл — LLM может дать финальный текстовый ответ
 
    return history
 
 
# ─────────────────────────────────────────
#  Точка входа
# ─────────────────────────────────────────
 
def main():
    print("╔══════════════════════════════════════════╗")
    print("║   РЕДАКТОР ИЗОБРАЖЕНИЙ — AI-управление   ║")
    print("╚══════════════════════════════════════════╝")
    print("Просто пишите, что нужно сделать с картинкой.")
    print("Примеры: «повернуть на 90», «сделать ч/б в стиле плёнки»,")
    print("         «отразить по горизонтали», «размыть сильно»,")
    print("         «поверни на 45 и сделай инфракрасное ч/б»")
    print("Введите «выход» или «сохранить» для завершения.\n")
 
    image_path = input("Файл изображения (например, photo.jpg): ").strip()
    try:
        processor = ImageProcessor(image_path)
        print(f"✅ Загружено: {image_path}  |  {processor.get_info()}\n")
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        sys.exit(1)
 
    history = []  # История разговора для multi-turn диалога
 
    while True:
        try:
            user_input = input("\nВы: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Выход.")
            break
 
        if not user_input:
            continue
 
        lower = user_input.lower()
 
        # Явные команды выхода/сохранения
        if lower in ("выход", "exit", "quit", "q"):
            processor.save_result("result.jpg")
            print("💾 Сохранено в result.jpg. Выход.")
            break
        if lower in ("сохранить", "save"):
            processor.save_result("result.jpg")
            print("💾 Сохранено в result.jpg.")
            continue
 
        # Всё остальное — через нейросеть
        history = process_command(user_input, processor, history)
 
 
if __name__ == "__main__":
    main()
 