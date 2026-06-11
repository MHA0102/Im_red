import cv2
import json
import sys
import numpy as np
from openai import OpenAI
 
client = OpenAI(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio"
)
 
class ImageProcessor:
    def __init__(self, image_path):
        self.image_path = image_path
        self.img = cv2.imread(image_path)
        if self.img is None:
            raise FileNotFoundError(f"Image not found at {image_path}")
        self.is_grayscale = False
 
    def rotate_image(self, angle: int):
        if angle == 90:
            self.img = cv2.rotate(self.img, cv2.ROTATE_90_CLOCKWISE)
        elif angle == 180:
            self.img = cv2.rotate(self.img, cv2.ROTATE_180)
        elif angle == 270:
            self.img = cv2.rotate(self.img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        else:
            h, w = self.img.shape[:2]
            cx, cy = w // 2, h // 2
            M = cv2.getRotationMatrix2D((cx, cy), -angle, 1.0)
            cos = abs(M[0, 0])
            sin = abs(M[0, 1])
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
        return "Размер изображения изменен"
 
    def flip_image(self, direction: str):
        if direction == "horizontal":
            self.img = cv2.flip(self.img, 1)
        elif direction == "vertical":
            self.img = cv2.flip(self.img, 0)
        return "Изображение зеркально отражено"
 
    def blur_image(self, kernel_size: int):
        if kernel_size % 2 == 0:
            kernel_size += 1
        self.img = cv2.GaussianBlur(self.img, (kernel_size, kernel_size), 0)
        return "Изображение размыто"
 
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
            l_eq = clahe.apply(l)
            lab_eq = cv2.merge([l_eq, a, b_ch])
            enhanced = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)
            gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
 
        elif mode == "soft":
            blurred = cv2.GaussianBlur(self.img, (3, 3), 0)
            gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
            gray = cv2.addWeighted(gray, 0.85, np.full_like(gray, 255), 0.15, 0)
 
        elif mode == "film":
            gray_f = 0.27 * r + 0.53 * g + 0.20 * b
            gray = np.clip(gray_f, 0, 255).astype(np.uint8)
            noise = np.random.normal(0, 4, gray.shape).astype(np.int16)
            gray = np.clip(gray.astype(np.int16) + noise, 0, 255).astype(np.uint8)
 
        elif mode == "infrared":
            gray_f = 0.07 * r + 0.72 * g + 0.21 * b
            gray = np.clip(gray_f, 0, 255).astype(np.uint8)
            gray = cv2.equalizeHist(gray)
 
        else:
            gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
 
        self.img = gray
        self.is_grayscale = True
        return "Изображение переведено в ч/б"
 
    def save_result(self, output_path="result.jpg"):
        cv2.imwrite(output_path, self.img)
 
 
tools = [
    {
        "type": "function",
        "function": {
            "name": "rotate_image",
            "description": "Повернуть изображение на заданный угол.",
            "parameters": {
                "type": "object",
                "properties": {
                    "angle": {"type": "integer", "description": "Угол поворота в градусах"}
                },
                "required": ["angle"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "resize_image",
            "description": "Изменить разрешение изображения.",
            "parameters": {
                "type": "object",
                "properties": {
                    "width": {"type": "integer"},
                    "height": {"type": "integer"}
                },
                "required": ["width", "height"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "flip_image",
            "description": "Отразить изображение.",
            "parameters": {
                "type": "object",
                "properties": {
                    "direction": {"type": "string", "enum": ["horizontal", "vertical"]}
                },
                "required": ["direction"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "blur_image",
            "description": "Размыть изображение.",
            "parameters": {
                "type": "object",
                "properties": {
                    "kernel_size": {"type": "integer"}
                },
                "required": ["kernel_size"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "convert_to_grayscale",
            "description": "Сделать изображение черно-белым.",
            "parameters": {
                "type": "object",
                "properties": {
                    "mode": {"type": "string", "enum": ["standard", "hc", "soft", "film", "infrared"]}
                },
                "required": ["mode"]
            }
        }
    }
]
 
def ask_llm(prompt, processor):
    response = client.chat.completions.create(
        model="local-model",
        messages=[{"role": "user", "content": prompt}],
        tools=tools,
        tool_choice="auto",
        temperature=0.0
    )
    message = response.choices[0].message
    if not message.tool_calls:
        print("Нейросеть не смогла распознать параметры.")
        return
    for tool_call in message.tool_calls:
        func_name = tool_call.function.name
        args = json.loads(tool_call.function.arguments)
        if hasattr(processor, func_name):
            result = getattr(processor, func_name)(**args)
            print(f"✅ Выполнено: {result}")
 
 
def submenu(title, options, descriptions=None):
    print(f"\n  {title}")
    for i, opt in enumerate(options, 1):
        desc = f"  — {descriptions[i-1]}" if descriptions else ""
        print(f"  {i}. {opt}{desc}")
    choice = input("  Ваш выбор: ").strip()
    if choice.isdigit() and 1 <= int(choice) <= len(options):
        return int(choice) - 1
    print("  ⚠️  Неверный ввод.")
    return None
 
 
if __name__ == "__main__":
    print("╔══════════════════════════════════════╗")
    print("║    РЕДАКТОР ИЗОБРАЖЕНИЙ (AI-версия)  ║")
    print("╚══════════════════════════════════════╝")
    image_path = input("Введите имя файла картинки (например, photo.jpg): ").strip()
 
    try:
        processor = ImageProcessor(image_path)
        print(f"✅ Файл '{image_path}' успешно загружен.\n")
    except Exception as e:
        print(f"❌ Ошибка при открытии картинки: {e}")
        sys.exit(1)
 
    MAIN_MENU = [
        "Повернуть изображение",
        "Сделать чёрно-белым",
        "Отразить зеркально",
        "Размыть",
        "Изменить размер (разрешение)",
        "Сохранить результат и выйти",
    ]
 
    ANGLES = [30, 45, 60, 90, 120, 135, 150, 180, 270, 0]
    ANGLE_LABELS = ["30°","45°","60°","90°","120°","135°","150°","180°","270°","Свой угол"]
 
    BW_MODES = ["standard", "hc", "soft", "film", "infrared"]
    BW_LABELS = [
        "Обычный",
        "Высококонтрастный",
        "Мягкий (плёночный)",
        "Имитация плёнки Kodak",
        "Инфракрасный эффект",
    ]
    BW_DESCS = [
        "стандартный перевод в grayscale",
        "CLAHE — резкие тени и света",
        "мягкие тона, приятный свет",
        "тёплые веса каналов + зерно",
        "небо тёмное, листья светлые",
    ]
 
    while True:
        print("\n┌─────────────────────────────────────┐")
        print("│         МЕНЮ УПРАВЛЕНИЯ             │")
        print("├─────────────────────────────────────┤")
        for i, item in enumerate(MAIN_MENU, 1):
            print(f"│  {i}. {item:<35}│")
        print("└─────────────────────────────────────┘")
 
        choice = input("Выберите действие (1-6): ").strip()
 
        if choice == "1":
            idx = submenu("Выберите угол поворота:", ANGLE_LABELS)
            if idx is not None:
                angle = ANGLES[idx]
                if angle == 0:
                    try:
                        angle = int(input("  Введите угол (0-360): ").strip())
                    except ValueError:
                        print("  ⚠️  Неверный угол.")
                        continue
                ask_llm(f"повернуть изображение на {angle} градусов", processor)
 
        elif choice == "2":
            idx = submenu("Выберите режим ч/б:", BW_LABELS, BW_DESCS)
            if idx is not None:
                mode = BW_MODES[idx]
                ask_llm(f"сделать изображение черно-белым в режиме {mode}", processor)
 
        elif choice == "3":
            idx = submenu("Как отразить?", ["По горизонтали (влево-вправо)", "По вертикали (вверх-вниз)"])
            if idx is not None:
                direction_ru = ["горизонтально", "вертикально"][idx]
                ask_llm(f"отрази {direction_ru}", processor)
 
        elif choice == "4":
            idx = submenu("Степень размытия?", ["Слабое (5)", "Среднее (15)", "Сильное (31)"])
            if idx is not None:
                kernel = [5, 15, 31][idx]
                ask_llm(f"размыть изображение с размером ядра {kernel}", processor)
 
        elif choice == "5":
            SIZES = [
                ("320 × 240  (маленький)", 320, 240),
                ("640 × 480  (средний)",   640, 480),
                ("800 × 600  (большой)",   800, 600),
                ("1280 × 720 (HD)",        1280, 720),
                ("1920 × 1080 (Full HD)", 1920, 1080),
                ("Свой размер",            None, None),
            ]
            idx = submenu("Выберите новый размер:", [s[0] for s in SIZES])
            if idx is not None:
                _, w, h = SIZES[idx]
                if w is None:
                    try:
                        w = int(input("  Ширина (px): ").strip())
                        h = int(input("  Высота (px): ").strip())
                    except ValueError:
                        print("  ⚠️  Неверный ввод размера.")
                        continue
                ask_llm(f"измени размер на {w}x{h}", processor)
 
        elif choice == "6":
            processor.save_result("result.jpg")
            print("💾 Результат сохранён в файл result.jpg.")
            print("👋 Выход из программы.")
            break
 
        else:
            print("⚠️  Неверный ввод, выберите число от 1 до 6.")
 