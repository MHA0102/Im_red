import cv2
import json
import sys
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import threading
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio"
)

# ── Цвета ──────────────────────────────────────────────────────────────────
BG        = "#0f0f13"
PANEL     = "#1a1a24"
CARD      = "#22223a"
ACCENT    = "#7c6af7"
ACCENT2   = "#a78bfa"
SUCCESS   = "#4ade80"
WARNING   = "#facc15"
DANGER    = "#f87171"
TEXT      = "#e2e8f0"
MUTED     = "#64748b"
BORDER    = "#2e2e4a"
BTN_HOVER = "#6d5ce6"

# ── ImageProcessor ─────────────────────────────────────────────────────────
class ImageProcessor:
    def __init__(self, image_path):
        self.image_path = image_path
        self.img = cv2.imread(image_path)
        if self.img is None:
            raise FileNotFoundError(f"Файл не найден: {image_path}")
        self.is_grayscale = False
        self._original = self.img.copy()   # оригинал — никогда не меняется
        self._history  = []                # стек состояний для undo

    def _save_state(self):
        """Сохраняет текущее состояние перед изменением."""
        self._history.append((self.img.copy(), self.is_grayscale))

    def undo(self):
        """Откатывает последнее действие."""
        if not self._history:
            return False
        self.img, self.is_grayscale = self._history.pop()
        return True

    def reset(self):
        """Возвращает оригинальное изображение."""
        self._history.clear()
        self.img = self._original.copy()
        self.is_grayscale = False

    def rotate_image(self, angle: int):
        self._save_state()
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
            cos, sin = abs(M[0,0]), abs(M[0,1])
            nw, nh = int(h*sin+w*cos), int(h*cos+w*sin)
            M[0,2] += nw/2 - cx; M[1,2] += nh/2 - cy
            self.img = cv2.warpAffine(self.img, M, (nw, nh),
                                      borderMode=cv2.BORDER_CONSTANT,
                                      borderValue=(0,0,0))
        return f"Повернуто на {angle}°"

    def resize_image(self, width: int, height: int):
        self._save_state()
        self.img = cv2.resize(self.img, (width, height))
        return f"Размер изменён → {width}×{height}"

    def flip_image(self, direction: str):
        self._save_state()
        self.img = cv2.flip(self.img, 1 if direction == "horizontal" else 0)
        return "Отражено " + ("по горизонтали" if direction == "horizontal" else "по вертикали")

    def blur_image(self, kernel_size: int):
        self._save_state()
        if kernel_size % 2 == 0: kernel_size += 1
        self.img = cv2.GaussianBlur(self.img, (kernel_size, kernel_size), 0)
        return f"Размытие (ядро {kernel_size})"

    def convert_to_grayscale(self, mode: str = "standard"):
        self._save_state()
        if self.is_grayscale or len(self.img.shape) == 2:
            self.img = cv2.cvtColor(self.img, cv2.COLOR_GRAY2BGR)
            self.is_grayscale = False
        b, g, r = cv2.split(self.img.astype(np.float32))
        if mode == "standard":
            gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        elif mode == "hc":
            lab = cv2.cvtColor(self.img, cv2.COLOR_BGR2LAB)
            l, a, b_ch = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            gray = cv2.cvtColor(cv2.cvtColor(cv2.merge([clahe.apply(l), a, b_ch]),
                                              cv2.COLOR_LAB2BGR), cv2.COLOR_BGR2GRAY)
        elif mode == "soft":
            blurred = cv2.GaussianBlur(self.img, (3,3), 0)
            gray = cv2.addWeighted(cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY),
                                   0.85, np.full_like(blurred[:,:,0], 255), 0.15, 0)
        elif mode == "film":
            gray_f = 0.27*r + 0.53*g + 0.20*b
            gray = np.clip(gray_f, 0, 255).astype(np.uint8)
            noise = np.random.normal(0, 4, gray.shape).astype(np.int16)
            gray = np.clip(gray.astype(np.int16)+noise, 0, 255).astype(np.uint8)
        elif mode == "infrared":
            gray_f = 0.07*r + 0.72*g + 0.21*b
            gray = cv2.equalizeHist(np.clip(gray_f, 0, 255).astype(np.uint8))
        else:
            gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        self.img = gray
        self.is_grayscale = True
        names = {"standard":"Обычный","hc":"Высококонтрастный","soft":"Мягкий",
                 "film":"Kodak T-MAX","infrared":"Инфракрасный"}
        return f"Ч/б — {names.get(mode, mode)}"

    def save_result(self, output_path="result.jpg"):
        cv2.imwrite(output_path, self.img)
        return output_path

    def get_pil(self, max_w=700, max_h=500):
        img = self.img.copy()
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        scale = min(max_w/w, max_h/h, 1.0)
        if scale < 1.0:
            img = cv2.resize(img, (int(w*scale), int(h*scale)))
        return Image.fromarray(img)

# ── Tools для LLM ──────────────────────────────────────────────────────────
TOOLS = [
    {"type":"function","function":{"name":"rotate_image","description":"Повернуть изображение.",
     "parameters":{"type":"object","properties":{"angle":{"type":"integer"}},"required":["angle"]}}},
    {"type":"function","function":{"name":"resize_image","description":"Изменить размер.",
     "parameters":{"type":"object","properties":{"width":{"type":"integer"},"height":{"type":"integer"}},"required":["width","height"]}}},
    {"type":"function","function":{"name":"flip_image","description":"Отразить изображение.",
     "parameters":{"type":"object","properties":{"direction":{"type":"string","enum":["horizontal","vertical"]}},"required":["direction"]}}},
    {"type":"function","function":{"name":"blur_image","description":"Размыть изображение.",
     "parameters":{"type":"object","properties":{"kernel_size":{"type":"integer"}},"required":["kernel_size"]}}},
    {"type":"function","function":{"name":"convert_to_grayscale","description":"Сделать ч/б.",
     "parameters":{"type":"object","properties":{"mode":{"type":"string","enum":["standard","hc","soft","film","infrared"]}},"required":["mode"]}}},
]

def ask_llm(prompt, processor):
    response = client.chat.completions.create(
        model="local-model",
        messages=[{"role":"user","content":prompt}],
        tools=TOOLS, tool_choice="auto", temperature=0.0
    )
    message = response.choices[0].message
    results = []
    if not message.tool_calls:
        return ["Нейросеть не распознала команду"]
    for tc in message.tool_calls:
        func_name = tc.function.name
        args = json.loads(tc.function.arguments)
        if hasattr(processor, func_name):
            result = getattr(processor, func_name)(**args)
            results.append(result)
    return results

# ── Главное окно ───────────────────────────────────────────────────────────
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("AI Image Editor")
        self.geometry("1200x780")
        self.minsize(900, 600)
        self.configure(bg=BG)
        self.processor = None
        self._build()

    def _build(self):
        # ── Заголовок
        header = tk.Frame(self, bg=PANEL, height=56)
        header.pack(fill="x")
        header.pack_propagate(False)
        tk.Label(header, text="✦  AI Image Editor", font=("Segoe UI", 15, "bold"),
                 bg=PANEL, fg=ACCENT2).pack(side="left", padx=24, pady=14)
        tk.Label(header, text="Нейросеть обрабатывает каждое действие",
                 font=("Segoe UI", 9), bg=PANEL, fg=MUTED).pack(side="left", pady=14)

        # ── Тело
        body = tk.Frame(self, bg=BG)
        body.pack(fill="both", expand=True, padx=0, pady=0)

        # Левая панель инструментов — со скроллом
        left_outer = tk.Frame(body, bg=PANEL, width=290)
        left_outer.pack(side="left", fill="y")
        left_outer.pack_propagate(False)

        left_canvas = tk.Canvas(left_outer, bg=PANEL, highlightthickness=0,
                                width=270)
        left_canvas.pack(side="left", fill="both", expand=True)

        left_scroll = tk.Scrollbar(left_outer, orient="vertical",
                                   command=left_canvas.yview)
        left_scroll.pack(side="right", fill="y")
        left_canvas.configure(yscrollcommand=left_scroll.set)

        left = tk.Frame(left_canvas, bg=PANEL)
        left_win = left_canvas.create_window((0, 0), window=left, anchor="nw")

        def _on_frame_configure(e):
            left_canvas.configure(scrollregion=left_canvas.bbox("all"))
        def _on_canvas_configure(e):
            left_canvas.itemconfig(left_win, width=e.width)
        def _on_mousewheel(e):
            left_canvas.yview_scroll(int(-1*(e.delta/120)), "units")

        left.bind("<Configure>", _on_frame_configure)
        left_canvas.bind("<Configure>", _on_canvas_configure)
        left_canvas.bind_all("<MouseWheel>", _on_mousewheel)

        self._build_tools(left)

        # Центр — превью
        center = tk.Frame(body, bg=BG)
        center.pack(side="left", fill="both", expand=True, padx=0)
        self._build_preview(center)

        # Правая панель — лог
        right = tk.Frame(body, bg=PANEL, width=240)
        right.pack(side="right", fill="y")
        right.pack_propagate(False)
        self._build_log(right)

    # ── Левая панель ───────────────────────────────────────────────────────
    def _build_tools(self, parent):
        tk.Label(parent, text="ИНСТРУМЕНТЫ", font=("Segoe UI", 9, "bold"),
                 bg=PANEL, fg=MUTED).pack(anchor="w", padx=20, pady=(18,8))

        # Открыть файл
        self._btn(parent, "📂  Открыть изображение", self._open_file, accent=True)

        # Undo / Reset
        frm_ur = tk.Frame(parent, bg=PANEL)
        frm_ur.pack(fill="x", padx=16, pady=(4,0))
        b_undo = tk.Button(frm_ur, text="↩ Отменить", command=self._undo,
                           bg=CARD, fg=TEXT, relief="flat", cursor="hand2",
                           font=("Segoe UI", 10), padx=10, pady=8, bd=0,
                           activebackground=BTN_HOVER, activeforeground="#fff")
        b_undo.pack(side="left", expand=True, fill="x", padx=(0,3))
        b_undo.bind("<Enter>", lambda e: b_undo.config(bg=BTN_HOVER))
        b_undo.bind("<Leave>", lambda e: b_undo.config(bg=CARD))

        b_reset = tk.Button(frm_ur, text="🔁 Оригинал", command=self._reset,
                            bg=CARD, fg=DANGER, relief="flat", cursor="hand2",
                            font=("Segoe UI", 10), padx=10, pady=8, bd=0,
                            activebackground="#7f1d1d", activeforeground="#fff")
        b_reset.pack(side="left", expand=True, fill="x", padx=(3,0))
        b_reset.bind("<Enter>", lambda e: b_reset.config(bg="#7f1d1d", fg="#fff"))
        b_reset.bind("<Leave>", lambda e: b_reset.config(bg=CARD, fg=DANGER))

        self._divider(parent)

        # Поворот
        self._section(parent, "🔄  Поворот")
        frm = tk.Frame(parent, bg=PANEL)
        frm.pack(fill="x", padx=16, pady=(0,4))
        for angle in [90, 180, 270]:
            self._small_btn(frm, f"{angle}°", lambda a=angle: self._action(f"повернуть на {a} градусов"))
        self._angle_row(parent)
        self._divider(parent)

        # Ч/б
        self._section(parent, "🎞️  Чёрно-белый")
        modes = [("Обычный","standard"),("Высококонтрастный","hc"),
                 ("Мягкий","soft"),("Kodak T-MAX","film"),("Инфракрасный","infrared")]
        for label, mode in modes:
            self._btn(parent, label, lambda m=mode: self._action(f"сделать чб в режиме {m}"))
        self._divider(parent)

        # Отражение
        self._section(parent, "🪞  Отражение")
        frm2 = tk.Frame(parent, bg=PANEL)
        frm2.pack(fill="x", padx=16, pady=(0,4))
        self._small_btn(frm2, "↔ Гориз.", lambda: self._action("отрази горизонтально"))
        self._small_btn(frm2, "↕ Верт.", lambda: self._action("отрази вертикально"))
        self._divider(parent)

        # Размытие
        self._section(parent, "💧  Размытие")
        frm3 = tk.Frame(parent, bg=PANEL)
        frm3.pack(fill="x", padx=16, pady=(0,4))
        for label, k in [("Слабое","5"),("Среднее","15"),("Сильное","31")]:
            self._small_btn(frm3, label, lambda k=k: self._action(f"размыть с ядром {k}"))
        self._divider(parent)

        # Размер
        self._section(parent, "📐  Размер")
        sizes = [("640×480","640 480"),("1280×720","1280 720"),("1920×1080","1920 1080"),("1080×1350  Portrait","1080 1350"),("1080×1920  Stories","1080 1920")]
        for label, wh in sizes:
            w, h = wh.split()
            self._btn(parent, label, lambda w=w,h=h: self._action(f"измени размер на {w}x{h}"))
        self._custom_size(parent)
        self._divider(parent)

        # Сохранить
        self._btn(parent, "💾  Сохранить", self._save, accent=True)

    def _section(self, parent, text):
        tk.Label(parent, text=text, font=("Segoe UI", 10, "bold"),
                 bg=PANEL, fg=TEXT).pack(anchor="w", padx=20, pady=(12,5))

    def _divider(self, parent):
        tk.Frame(parent, bg=BORDER, height=1).pack(fill="x", padx=16, pady=8)

    def _btn(self, parent, text, cmd, accent=False):
        bg = ACCENT if accent else CARD
        fg = "#fff" if accent else TEXT
        b = tk.Button(parent, text=text, command=cmd,
                      bg=bg, fg=fg, relief="flat", cursor="hand2",
                      font=("Segoe UI", 10), anchor="w",
                      padx=14, pady=10, bd=0,
                      activebackground=BTN_HOVER, activeforeground="#fff")
        b.pack(fill="x", padx=16, pady=3)
        b.bind("<Enter>", lambda e: b.config(bg=BTN_HOVER if accent else "#2d2d4a"))
        b.bind("<Leave>", lambda e: b.config(bg=bg))

    def _small_btn(self, parent, text, cmd):
        b = tk.Button(parent, text=text, command=cmd,
                      bg=CARD, fg=TEXT, relief="flat", cursor="hand2",
                      font=("Segoe UI", 10), padx=12, pady=8, bd=0,
                      activebackground=BTN_HOVER, activeforeground="#fff")
        b.pack(side="left", padx=3, pady=3)
        b.bind("<Enter>", lambda e: b.config(bg=BTN_HOVER))
        b.bind("<Leave>", lambda e: b.config(bg=CARD))

    def _angle_row(self, parent):
        frm = tk.Frame(parent, bg=PANEL)
        frm.pack(fill="x", padx=16, pady=(0,4))
        self.angle_var = tk.StringVar(value="45")
        e = tk.Entry(frm, textvariable=self.angle_var, width=5,
                     bg=CARD, fg=TEXT, insertbackground=TEXT,
                     relief="flat", font=("Segoe UI", 10))
        e.pack(side="left", padx=(0,6), ipady=7)
        b = tk.Button(frm, text="Повернуть", command=self._custom_rotate,
                      bg=CARD, fg=TEXT, relief="flat", cursor="hand2",
                      font=("Segoe UI", 10), padx=12, pady=7, bd=0,
                      activebackground=BTN_HOVER, activeforeground="#fff")
        b.pack(side="left")
        b.bind("<Enter>", lambda e: b.config(bg=BTN_HOVER))
        b.bind("<Leave>", lambda e: b.config(bg=CARD))

    def _custom_size(self, parent):
        frm = tk.Frame(parent, bg=PANEL)
        frm.pack(fill="x", padx=16, pady=(4,2))
        self.cw = tk.Entry(frm, width=5, bg=CARD, fg=TEXT, insertbackground=TEXT,
                           relief="flat", font=("Segoe UI", 10))
        self.cw.insert(0, "800")
        self.cw.pack(side="left", ipady=7, padx=(0,4))
        tk.Label(frm, text="×", bg=PANEL, fg=MUTED, font=("Segoe UI", 10)).pack(side="left")
        self.ch = tk.Entry(frm, width=5, bg=CARD, fg=TEXT, insertbackground=TEXT,
                           relief="flat", font=("Segoe UI", 10))
        self.ch.insert(0, "600")
        self.ch.pack(side="left", ipady=7, padx=(4,6))
        b = tk.Button(frm, text="OK", command=self._custom_resize,
                      bg=CARD, fg=TEXT, relief="flat", cursor="hand2",
                      font=("Segoe UI", 10), padx=12, pady=7, bd=0,
                      activebackground=BTN_HOVER, activeforeground="#fff")
        b.pack(side="left")
        b.bind("<Enter>", lambda e: b.config(bg=BTN_HOVER))
        b.bind("<Leave>", lambda e: b.config(bg=CARD))

    # ── Превью ─────────────────────────────────────────────────────────────
    def _build_preview(self, parent):
        # Строка пути файла
        top = tk.Frame(parent, bg=CARD, height=38)
        top.pack(fill="x", padx=12, pady=(12,0))
        top.pack_propagate(False)
        self.path_label = tk.Label(top, text="Файл не открыт",
                                   font=("Segoe UI", 9), bg=CARD, fg=MUTED)
        self.path_label.pack(side="left", padx=12, pady=8)
        self.info_label = tk.Label(top, text="", font=("Segoe UI", 8),
                                   bg=CARD, fg=MUTED)
        self.info_label.pack(side="right", padx=12, pady=8)

        # Холст превью
        self.canvas = tk.Canvas(parent, bg="#0a0a10", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True, padx=12, pady=8)
        self.canvas.bind("<Configure>", lambda e: self._refresh_preview())

        # Плейсхолдер
        self._placeholder()

    def _placeholder(self):
        self.canvas.delete("all")
        w = self.canvas.winfo_width() or 700
        h = self.canvas.winfo_height() or 500
        self.canvas.create_text(w//2, h//2, text="📂\n\nОткройте изображение",
                                 font=("Segoe UI", 13), fill=MUTED, justify="center")

    # ── Лог ────────────────────────────────────────────────────────────────
    def _build_log(self, parent):
        tk.Label(parent, text="ИСТОРИЯ", font=("Segoe UI", 8, "bold"),
                 bg=PANEL, fg=MUTED).pack(anchor="w", padx=16, pady=(18,6))
        self.log_frame = tk.Frame(parent, bg=PANEL)
        self.log_frame.pack(fill="both", expand=True, padx=8)
        sb = tk.Scrollbar(self.log_frame)
        sb.pack(side="right", fill="y")
        self.log_box = tk.Text(self.log_frame, bg=PANEL, fg=TEXT,
                               font=("Segoe UI", 8), relief="flat",
                               wrap="word", state="disabled",
                               yscrollcommand=sb.set, bd=0,
                               selectbackground=ACCENT)
        self.log_box.pack(fill="both", expand=True)
        sb.config(command=self.log_box.yview)
        self.log_box.tag_config("ok",  foreground=SUCCESS)
        self.log_box.tag_config("err", foreground=DANGER)
        self.log_box.tag_config("dim", foreground=MUTED)

        # Статус
        self.status_var = tk.StringVar(value="Готов")
        self.status_bar = tk.Label(parent, textvariable=self.status_var,
                                   font=("Segoe UI", 8), bg=CARD, fg=MUTED,
                                   anchor="w")
        self.status_bar.pack(fill="x", padx=8, pady=(4,12), ipady=4)

    def _log(self, text, tag="ok"):
        self.log_box.config(state="normal")
        self.log_box.insert("end", f"• {text}\n", tag)
        self.log_box.see("end")
        self.log_box.config(state="disabled")

    def _status(self, text, color=MUTED):
        self.status_var.set(text)
        self.status_bar.config(fg=color)

    # ── Действия ───────────────────────────────────────────────────────────
    def _undo(self):
        if not self.processor:
            return
        if self.processor.undo():
            self._refresh_preview()
            self._update_info()
            self._log("Отменено последнее действие", "dim")
            self._status("Отменено", WARNING)
        else:
            self._log("Нечего отменять", "dim")

    def _reset(self):
        if not self.processor:
            return
        if messagebox.askyesno("Сброс", "Вернуть оригинальное изображение?\nВся история будет удалена."):
            self.processor.reset()
            self._refresh_preview()
            self._update_info()
            self._log("Сброс до оригинала", "dim")
            self._status("Сброшено до оригинала", WARNING)

    def _open_file(self):
        path = filedialog.askopenfilename(
            filetypes=[("Изображения","*.jpg *.jpeg *.png *.bmp *.webp *.tiff"),
                       ("Все файлы","*.*")])
        if not path: return
        try:
            self.processor = ImageProcessor(path)
            self.path_label.config(text=path.split("/")[-1].split("\\")[-1], fg=TEXT)
            self._update_info()
            self._refresh_preview()
            self._log(f"Открыт: {path.split('/')[-1].split(chr(92))[-1]}", "ok")
            self._status("Файл загружен", SUCCESS)
        except Exception as e:
            messagebox.showerror("Ошибка", str(e))
            self._log(str(e), "err")

    def _action(self, prompt):
        if not self.processor:
            messagebox.showwarning("Нет файла", "Сначала откройте изображение.")
            return
        self._status("⏳ Нейросеть обрабатывает...", WARNING)
        self.update()
        def run():
            try:
                results = ask_llm(prompt, self.processor)
                self.after(0, lambda: self._on_done(results))
            except Exception as e:
                self.after(0, lambda: self._on_error(str(e)))
        threading.Thread(target=run, daemon=True).start()

    def _on_done(self, results):
        for r in results:
            self._log(r, "ok")
        self._refresh_preview()
        self._update_info()
        self._status("Готов", SUCCESS)

    def _on_error(self, err):
        self._log(err, "err")
        self._status("Ошибка", DANGER)

    def _custom_rotate(self):
        try:
            angle = int(self.angle_var.get())
            self._action(f"повернуть на {angle} градусов")
        except ValueError:
            messagebox.showwarning("Ошибка", "Введите целое число.")

    def _custom_resize(self):
        try:
            w, h = int(self.cw.get()), int(self.ch.get())
            self._action(f"измени размер на {w}x{h}")
        except ValueError:
            messagebox.showwarning("Ошибка", "Введите целые числа.")

    def _save(self):
        if not self.processor:
            messagebox.showwarning("Нет файла", "Нечего сохранять.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".jpg",
            filetypes=[("JPEG","*.jpg"),("PNG","*.png"),("BMP","*.bmp")])
        if not path: return
        self.processor.save_result(path)
        self._log(f"Сохранено: {path.split('/')[-1].split(chr(92))[-1]}", "ok")
        self._status("Сохранено ✓", SUCCESS)

    def _refresh_preview(self):
        if not self.processor:
            self._placeholder()
            return
        self.canvas.update_idletasks()
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw < 10 or ch < 10:
            return
        pil = self.processor.get_pil(cw - 20, ch - 20)
        self._tk_img = ImageTk.PhotoImage(pil)
        self.canvas.delete("all")
        self.canvas.create_image(cw//2, ch//2, anchor="center", image=self._tk_img)

    def _update_info(self):
        if not self.processor: return
        h, w = self.processor.img.shape[:2]
        ch = 1 if len(self.processor.img.shape) == 2 else self.processor.img.shape[2]
        mode = "Ч/Б" if ch == 1 else "RGB"
        self.info_label.config(text=f"{w} × {h}  •  {mode}", fg=MUTED)


if __name__ == "__main__":
    try:
        from PIL import Image, ImageTk
    except ImportError:
        print("Установите Pillow:  pip install Pillow")
        sys.exit(1)
    app = App()
    app.mainloop()
 
