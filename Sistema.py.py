import cv2
import pandas as pd
from datetime import datetime
import os
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
from PIL import Image, ImageTk
import threading
import queue
import time
from concurrent.futures import ThreadPoolExecutor


class SistemaReconhecimentoFacial:
    def __init__(self, root):
        self.root = root
        self.root.title("Sistema de Reconhecimento Facial")
        self.root.geometry("1200x700")
        self.root.configure(bg='#f0f0f0')

        # Variáveis de controle
        self.reconhecendo = False
        self.capturando = False
        self.webcam = None
        self.webcam_cadastro = None
        self.recognizer = None
        self.registros_hoje = set()  # Mudado de registrados_hoje para registros_hoje
        self.fila = queue.Queue()
        self.cache_nomes = {}

        # Configurações
        self.total_imagens_cadastro = 20
        self.ultima_captura = 0
        self.imagens_capturadas = 0
        self.camera_index = 0
        self.camera_index_cadastro = 0

        # Detector de faces
        try:
            self.detector_face = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao carregar detector: {e}")

        # Criar pastas necessárias
        os.makedirs("faces", exist_ok=True)
        os.makedirs("registros", exist_ok=True)
        os.makedirs("recognizer", exist_ok=True)

        self.setup_ui_simplificado()
        self.verificar_camera()
        self.atualizar_interface()
        self.atualizar_cache_nomes()
        self.carregar_registros_hoje()  # Carregar registros do dia atual

    def setup_ui_simplificado(self):
        """Interface simplificada e funcional"""
        # ========== MENU SUPERIOR ==========
        menu_frame = tk.Frame(self.root, bg='#2c3e50', height=50)
        menu_frame.pack(fill=tk.X)

        titulo = tk.Label(menu_frame, text="SISTEMA DE RECONHECIMENTO FACIAL",
                          font=('Arial', 16, 'bold'), bg='#2c3e50', fg='white')
        titulo.pack(pady=10)

        # ========== NOTBOOK (ABAS) ==========
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # --- ABA 1: CADASTRO ---
        self.tab_cadastro = tk.Frame(self.notebook, bg='white')
        self.notebook.add(self.tab_cadastro, text='📝 CADASTRO DE FUNCIONÁRIO')
        self.setup_aba_cadastro()

        # --- ABA 2: RECONHECIMENTO ---
        self.tab_reconhecimento = tk.Frame(self.notebook, bg='white')
        self.notebook.add(self.tab_reconhecimento, text='👁️ RECONHECIMENTO')
        self.setup_aba_reconhecimento()

        # --- ABA 3: TREINAMENTO ---
        self.tab_treinamento = tk.Frame(self.notebook, bg='white')
        self.notebook.add(self.tab_treinamento, text='⚙️ TREINAMENTO')
        self.setup_aba_treinamento()

        # ========== STATUS BAR ==========
        self.status_bar = tk.Label(self.root, text="✅ Sistema pronto",
                                   bd=1, relief=tk.SUNKEN, anchor=tk.W,
                                   font=('Arial', 9), bg='#ecf0f1')
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def setup_aba_cadastro(self):
        """Aba de cadastro SIMPLES e FUNCIONAL"""
        # Frame principal dividido em dois
        main_frame = tk.Frame(self.tab_cadastro, bg='white')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # ===== LADO ESQUERDO - FORMULÁRIO =====
        left_frame = tk.Frame(main_frame, bg='#ecf0f1', relief=tk.RIDGE, bd=2)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        # Título
        tk.Label(left_frame, text="CADASTRO DE FUNCIONÁRIO",
                 font=('Arial', 14, 'bold'), bg='#ecf0f1', fg='#2c3e50').pack(pady=20)

        # Frame para centralizar os campos
        form_frame = tk.Frame(left_frame, bg='#ecf0f1')
        form_frame.pack(expand=True)

        # Campo NOME
        tk.Label(form_frame, text="NOME COMPLETO:", font=('Arial', 11),
                 bg='#ecf0f1', fg='#34495e').pack(anchor=tk.W, pady=(10, 0))

        self.nome_entry = tk.Entry(form_frame, font=('Arial', 12), width=30,
                                   bg='white', relief=tk.SUNKEN, bd=2)
        self.nome_entry.pack(pady=(5, 15), ipady=5)

        # Campo ID
        tk.Label(form_frame, text="ID DO FUNCIONÁRIO:", font=('Arial', 11),
                 bg='#ecf0f1', fg='#34495e').pack(anchor=tk.W, pady=(0, 0))

        self.id_entry = tk.Entry(form_frame, font=('Arial', 12), width=30,
                                 bg='white', relief=tk.SUNKEN, bd=2)
        self.id_entry.pack(pady=(5, 20), ipady=5)

        # BOTÃO INICIAR CADASTRO - GRANDE E VISÍVEL
        self.btn_iniciar_cadastro = tk.Button(form_frame,
                                              text="📸 INICIAR CADASTRO",
                                              command=self.iniciar_cadastro,
                                              font=('Arial', 14, 'bold'),
                                              bg='#27ae60', fg='white',
                                              width=25, height=2,
                                              relief=tk.RAISED, bd=3,
                                              cursor='hand2')
        self.btn_iniciar_cadastro.pack(pady=20)

        # BOTÃO PARAR
        self.btn_parar_cadastro = tk.Button(form_frame,
                                            text="⏹ PARAR CADASTRO",
                                            command=self.parar_cadastro,
                                            font=('Arial', 12),
                                            bg='#e74c3c', fg='white',
                                            width=20, height=1,
                                            relief=tk.RAISED, bd=2,
                                            cursor='hand2',
                                            state='disabled')
        self.btn_parar_cadastro.pack(pady=10)

        # Status da câmera
        self.status_camera_label = tk.Label(left_frame,
                                            text="🔴 Verificando câmera...",
                                            font=('Arial', 11),
                                            bg='#ecf0f1', fg='#e67e22')
        self.status_camera_label.pack(pady=20)

        # ===== LADO DIREITO - CÂMERA E PROGRESSO =====
        right_frame = tk.Frame(main_frame, bg='#bdc3c7', relief=tk.RIDGE, bd=2)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Frame da câmera
        camera_container = tk.Frame(right_frame, bg='black', width=640, height=480)
        camera_container.pack(padx=10, pady=10)
        camera_container.pack_propagate(False)

        self.camera_cadastro_label = tk.Label(camera_container, bg='black')
        self.camera_cadastro_label.pack(fill=tk.BOTH, expand=True)

        # Progresso
        progress_frame = tk.Frame(right_frame, bg='#bdc3c7')
        progress_frame.pack(fill=tk.X, padx=10, pady=10)

        tk.Label(progress_frame, text="PROGRESSO:", font=('Arial', 11, 'bold'),
                 bg='#bdc3c7', fg='#2c3e50').pack(anchor=tk.W)

        self.progress_var = tk.IntVar()
        self.progress_bar = ttk.Progressbar(progress_frame,
                                            variable=self.progress_var,
                                            maximum=self.total_imagens_cadastro,
                                            length=400, mode='determinate')
        self.progress_bar.pack(fill=tk.X, pady=5)

        self.progress_label = tk.Label(progress_frame,
                                       text=f"0/{self.total_imagens_cadastro} imagens",
                                       font=('Arial', 10), bg='#bdc3c7', fg='#27ae60')
        self.progress_label.pack()

        # Instruções
        instrucoes = tk.Label(right_frame,
                              text="Dicas:\n• Posicione o rosto na área verde\n• Mantenha boa iluminação\n• Aguarde a captura automática",
                              font=('Arial', 10), bg='#bdc3c7', fg='#34495e',
                              justify=tk.LEFT)
        instrucoes.pack(pady=10)

    def setup_aba_reconhecimento(self):
        """Aba de reconhecimento simplificada"""
        main_frame = tk.Frame(self.tab_reconhecimento, bg='white')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Câmera
        camera_frame = tk.Frame(main_frame, bg='black', width=640, height=480)
        camera_frame.pack(side=tk.LEFT, padx=(0, 20))
        camera_frame.pack_propagate(False)

        self.camera_label = tk.Label(camera_frame, bg='black')
        self.camera_label.pack(fill=tk.BOTH, expand=True)

        # Controles
        control_frame = tk.Frame(main_frame, bg='#ecf0f1', relief=tk.RIDGE, bd=2)
        control_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        tk.Label(control_frame, text="RECONHECIMENTO",
                 font=('Arial', 14, 'bold'), bg='#ecf0f1', fg='#2c3e50').pack(pady=20)

        self.btn_iniciar_reconhecimento = tk.Button(control_frame,
                                                    text="▶ INICIAR RECONHECIMENTO",
                                                    command=self.iniciar_reconhecimento,
                                                    font=('Arial', 12, 'bold'),
                                                    bg='#27ae60', fg='white',
                                                    width=25, height=2,
                                                    relief=tk.RAISED, bd=3,
                                                    cursor='hand2')
        self.btn_iniciar_reconhecimento.pack(pady=10)

        self.btn_parar_reconhecimento = tk.Button(control_frame,
                                                  text="⏹ PARAR",
                                                  command=self.parar_reconhecimento,
                                                  font=('Arial', 12),
                                                  bg='#e74c3c', fg='white',
                                                  width=20, height=1,
                                                  relief=tk.RAISED, bd=2,
                                                  cursor='hand2',
                                                  state='disabled')
        self.btn_parar_reconhecimento.pack(pady=10)

        # Botão para limpar registros do dia
        self.btn_limpar_registros = tk.Button(control_frame,
                                              text="🗑 LIMPAR REGISTROS DO DIA",
                                              command=self.limpar_registros_hoje,
                                              font=('Arial', 10),
                                              bg='#f39c12', fg='white',
                                              width=20, height=1,
                                              relief=tk.RAISED, bd=2,
                                              cursor='hand2')
        self.btn_limpar_registros.pack(pady=5)

        # Lista de registros
        tk.Label(control_frame, text="ÚLTIMOS REGISTROS DE HOJE:",
                 font=('Arial', 11, 'bold'), bg='#ecf0f1', fg='#34495e').pack(pady=(30, 10))

        # Frame para a treeview com scrollbar
        tree_frame = tk.Frame(control_frame, bg='#ecf0f1')
        tree_frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(tree_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.tree_registros = ttk.Treeview(tree_frame,
                                           columns=('Hora', 'Nome', 'ID', 'Confiança'),
                                           show='headings', height=8,
                                           yscrollcommand=scrollbar.set)
        self.tree_registros.pack(fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.tree_registros.yview)

        self.tree_registros.heading('Hora', text='HORA')
        self.tree_registros.heading('Nome', text='NOME')
        self.tree_registros.heading('ID', text='ID')
        self.tree_registros.heading('Confiança', text='CONFIANÇA')

        # Configurar larguras das colunas
        self.tree_registros.column('Hora', width=80)
        self.tree_registros.column('Nome', width=150)
        self.tree_registros.column('ID', width=50)
        self.tree_registros.column('Confiança', width=80)

    def setup_aba_treinamento(self):
        """Aba de treinamento simplificada"""
        main_frame = tk.Frame(self.tab_treinamento, bg='white')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        info_frame = tk.Frame(main_frame, bg='#ecf0f1', relief=tk.RIDGE, bd=2)
        info_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        tk.Label(info_frame, text="TREINAMENTO DO MODELO",
                 font=('Arial', 16, 'bold'), bg='#ecf0f1', fg='#2c3e50').pack(pady=30)

        # Estatísticas
        stats_frame = tk.Frame(info_frame, bg='#ecf0f1')
        stats_frame.pack(pady=20)

        # Label para funcionários cadastrados
        self.label_funcionarios = tk.Label(stats_frame,
                                           text=f"Funcionários cadastrados: {len(self.cache_nomes)}",
                                           font=('Arial', 12), bg='#ecf0f1', fg='#34495e')
        self.label_funcionarios.pack(pady=5)

        # Label para total de imagens
        self.label_imagens = tk.Label(stats_frame,
                                      text=f"Total de imagens: {self.contar_total_imagens()}",
                                      font=('Arial', 12), bg='#ecf0f1', fg='#34495e')
        self.label_imagens.pack(pady=5)

        self.status_treinamento = tk.Label(info_frame,
                                           text="Clique em TREINAR para atualizar o modelo",
                                           font=('Arial', 11), bg='#ecf0f1', fg='#e67e22')
        self.status_treinamento.pack(pady=10)

        # Botão treinar
        self.btn_treinar = tk.Button(info_frame,
                                     text="🎯 TREINAR MODELO",
                                     command=self.treinar_modelo,
                                     font=('Arial', 14, 'bold'),
                                     bg='#3498db', fg='white',
                                     width=25, height=2,
                                     relief=tk.RAISED, bd=3,
                                     cursor='hand2')
        self.btn_treinar.pack(pady=20)

        # Log
        self.log_text = scrolledtext.ScrolledText(info_frame,
                                                  height=10, width=60,
                                                  font=('Courier', 10))
        self.log_text.pack(pady=20, padx=20, fill=tk.BOTH, expand=True)

    def verificar_camera(self):
        """Verifica a câmera de forma simples"""
        try:
            # Tentar diferentes índices de câmera
            indices_tentar = [0, 1, 2]
            for i in indices_tentar:
                cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
                if cap.isOpened():
                    # Testar se consegue ler um frame
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        self.camera_index = i
                        self.camera_index_cadastro = i
                        cap.release()
                        if hasattr(self, 'status_camera_label'):
                            self.status_camera_label.config(
                                text=f"🟢 CÂMERA {i} DISPONÍVEL", fg='#27ae60')
                        self.log(f"✅ Câmera detectada no índice {i}")
                        return True
                cap.release()

            if hasattr(self, 'status_camera_label'):
                self.status_camera_label.config(
                    text="🔴 CÂMERA NÃO ENCONTRADA", fg='#e74c3c')
            return False
        except Exception as e:
            if hasattr(self, 'status_camera_label'):
                self.status_camera_label.config(
                    text=f"🔴 ERRO NA CÂMERA", fg='#e74c3c')
            return False

    def carregar_registros_hoje(self):
        """Carrega os registros do dia atual para a interface"""
        try:
            self.registros_hoje.clear()

            # Limpar treeview
            if hasattr(self, 'tree_registros'):
                for item in self.tree_registros.get_children():
                    self.tree_registros.delete(item)

            # Carregar registros do arquivo CSV
            arquivo = "registros/presenca.csv"
            if os.path.exists(arquivo):
                df = pd.read_csv(arquivo)
                hoje = datetime.now().strftime('%Y-%m-%d')

                # Filtrar registros de hoje
                df_hoje = df[df['Data'] == hoje]

                # Adicionar à treeview e ao set de registros
                for _, row in df_hoje.iterrows():
                    self.registros_hoje.add(row['ID'])
                    self.tree_registros.insert('', 0, values=(
                        row['Hora'],
                        row['Nome'],
                        row['ID'],
                        row['Confianca']
                    ))

                self.log(f"📋 Carregados {len(df_hoje)} registros de hoje")
        except Exception as e:
            self.log(f"Erro ao carregar registros: {e}")

    def contar_total_imagens(self):
        """Conta o total de imagens disponíveis"""
        try:
            total = 0
            if os.path.exists("faces"):
                for pasta in os.listdir("faces"):
                    caminho = os.path.join("faces", pasta)
                    if os.path.isdir(caminho):
                        for arquivo in os.listdir(caminho):
                            if arquivo.lower().endswith(('.jpg', '.jpeg', '.png')):
                                total += 1
            return total
        except:
            return 0

    def iniciar_cadastro(self):
        """Inicia o cadastro de forma simples"""
        nome = self.nome_entry.get().strip()
        id_func = self.id_entry.get().strip()

        if not nome or not id_func:
            messagebox.showwarning("Atenção", "Preencha nome e ID do funcionário!")
            return

        # Verificar se ID é número
        try:
            id_func = int(id_func)
        except:
            messagebox.showerror("Erro", "ID deve ser um número!")
            return

        # Verificar câmera
        if not self.verificar_camera():
            messagebox.showerror("Erro", "Câmera não disponível!")
            return

        # Criar pasta
        self.pasta_destino = f"faces/{id_func}_{nome}"
        os.makedirs(self.pasta_destino, exist_ok=True)

        # Resetar variáveis
        self.imagens_capturadas = 0
        self.progress_var.set(0)
        self.progress_label.config(text=f"0/{self.total_imagens_cadastro} imagens")
        self.capturando = True

        # Atualizar interface
        self.btn_iniciar_cadastro.config(state='disabled', bg='#95a5a6')
        self.btn_parar_cadastro.config(state='normal', bg='#e74c3c')

        self.log(f"📸 Iniciando cadastro: {nome} (ID: {id_func})")

        # Iniciar captura
        self.thread_captura = threading.Thread(target=self.capturar_faces, daemon=True)
        self.thread_captura.start()

    def capturar_faces(self):
        """Captura faces de forma simples e rápida"""
        try:
            self.webcam_cadastro = cv2.VideoCapture(0)
            self.webcam_cadastro.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.webcam_cadastro.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

            if not self.webcam_cadastro.isOpened():
                self.fila.put(("erro", "Não foi possível abrir a câmera!"))
                return

            nome = self.nome_entry.get().strip()

            while self.capturando and self.imagens_capturadas < self.total_imagens_cadastro:
                ret, frame = self.webcam_cadastro.read()
                if not ret:
                    continue

                frame_processado = frame.copy()
                cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # Detectar faces
                faces = self.detector_face.detectMultiScale(
                    cinza, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

                # Desenhar retângulos
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame_processado, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Adicionar informações no frame
                cv2.putText(frame_processado, f"CADASTRO: {nome}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                cv2.putText(frame_processado, f"IMAGENS: {self.imagens_capturadas}/{self.total_imagens_cadastro}",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                # Capturar imagens
                if len(faces) > 0:
                    for (x, y, w, h) in faces:
                        if time.time() - self.ultima_captura > 0.3 and self.imagens_capturadas < self.total_imagens_cadastro:
                            rosto = frame[y:y + h, x:x + w]
                            rosto = cv2.resize(rosto, (200, 200))

                            self.imagens_capturadas += 1
                            cv2.imwrite(f"{self.pasta_destino}/{self.imagens_capturadas}.jpg", rosto)
                            self.ultima_captura = time.time()

                            # Atualizar progresso
                            self.progress_var.set(self.imagens_capturadas)
                            self.progress_label.config(
                                text=f"{self.imagens_capturadas}/{self.total_imagens_cadastro} imagens")

                            if self.imagens_capturadas >= self.total_imagens_cadastro:
                                break

                # Mostrar frame
                frame_rgb = cv2.cvtColor(frame_processado, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                img = img.resize((640, 480))
                imgtk = ImageTk.PhotoImage(image=img)
                self.camera_cadastro_label.config(image=imgtk)
                self.camera_cadastro_label.image = imgtk

                time.sleep(0.03)

            # Finalizar cadastro
            if self.imagens_capturadas > 0:
                self.fila.put(("concluido", f"Cadastro concluído! {self.imagens_capturadas} imagens salvas."))
                self.atualizar_cache_nomes()
            else:
                self.fila.put(("aviso", "Nenhuma imagem capturada."))

        except Exception as e:
            self.fila.put(("erro", f"Erro na captura: {str(e)}"))
        finally:
            if self.webcam_cadastro:
                self.webcam_cadastro.release()

    def iniciar_reconhecimento(self):
        """Inicia o reconhecimento facial"""
        if not os.path.exists("recognizer/trainer.yml"):
            messagebox.showerror("Erro", "Treine o modelo primeiro!")
            return

        # Carregar registros de hoje antes de iniciar
        self.carregar_registros_hoje()

        self.reconhecendo = True
        self.btn_iniciar_reconhecimento.config(state='disabled', bg='#95a5a6')
        self.btn_parar_reconhecimento.config(state='normal', bg='#e74c3c')

        # Carregar modelo
        self.carregar_modelo()

        # Iniciar captura
        self.thread_reconhecimento = threading.Thread(target=self.capturar_reconhecimento, daemon=True)
        self.thread_reconhecimento.start()

        self.log("👁️ Reconhecimento iniciado")

    def carregar_modelo(self):
        """Carrega o modelo de reconhecimento"""
        try:
            self.recognizer = cv2.face.LBPHFaceRecognizer_create()
            self.recognizer.read("recognizer/trainer.yml")
            self.log("✅ Modelo carregado")
        except Exception as e:
            self.fila.put(("erro", f"Erro ao carregar modelo: {e}"))

    def capturar_reconhecimento(self):
        """Captura e reconhece faces"""
        try:
            self.webcam = cv2.VideoCapture(0)
            self.webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

            if not self.webcam.isOpened():
                self.fila.put(("erro", "Não foi possível abrir a câmera!"))
                return

            while self.reconhecendo:
                ret, frame = self.webcam.read()
                if not ret:
                    continue

                cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.detector_face.detectMultiScale(
                    cinza, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                for (x, y, w, h) in faces:
                    if self.recognizer:
                        try:
                            id_pred, conf = self.recognizer.predict(cinza[y:y + h, x:x + w])
                            nome = self.cache_nomes.get(id_pred, "Desconhecido")

                            if conf < 80:  # Confiança boa (menor é melhor no LBPH)
                                cv2.rectangle(frame_rgb, (x, y), (x + w, y + h), (0, 255, 0), 2)
                                cv2.putText(frame_rgb, f"{nome}", (x, y - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                                cv2.putText(frame_rgb, f"{100 - conf:.0f}%", (x, y + h + 20),
                                            # Converter para porcentagem de confiança
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                                # Registrar presença (verificar se já registrou hoje)
                                if id_pred not in self.registros_hoje:
                                    self.fila.put(("registrar", (id_pred, nome, 100 - conf)))  # Passar porcentagem
                            else:
                                cv2.rectangle(frame_rgb, (x, y), (x + w, y + h), (0, 0, 255), 2)
                                cv2.putText(frame_rgb, "Desconhecido", (x, y - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                        except Exception as e:
                            cv2.rectangle(frame_rgb, (x, y), (x + w, y + h), (0, 0, 255), 2)
                            cv2.putText(frame_rgb, "Erro", (x, y - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                # Mostrar frame
                img = Image.fromarray(frame_rgb)
                img = img.resize((640, 480))
                imgtk = ImageTk.PhotoImage(image=img)
                self.camera_label.config(image=imgtk)
                self.camera_label.image = imgtk

                time.sleep(0.03)

        except Exception as e:
            self.log(f"Erro no reconhecimento: {e}")
        finally:
            if self.webcam:
                self.webcam.release()

    def treinar_modelo(self):
        """Treina o modelo de forma simples e eficiente"""

        def treinar():
            try:
                faces = []
                ids = []
                funcionarios = 0
                imagens_por_funcionario = {}

                self.status_treinamento.config(text="⏳ Carregando imagens...", fg='#e67e22')
                self.log("🔍 Iniciando treinamento...")

                if not os.path.exists("faces"):
                    self.fila.put(("erro_treinamento", "Pasta 'faces' não encontrada!"))
                    return

                # Listar todas as pastas de funcionários
                pastas = [p for p in os.listdir("faces") if os.path.isdir(os.path.join("faces", p))]

                if len(pastas) == 0:
                    self.fila.put(("erro_treinamento", "Nenhum funcionário cadastrado!"))
                    return

                self.log(f"📂 Encontradas {len(pastas)} pastas de funcionários")

                for pasta in pastas:
                    caminho = os.path.join("faces", pasta)
                    try:
                        # Extrair ID e nome da pasta
                        partes = pasta.split("_", 1)
                        if len(partes) != 2:
                            continue

                        id_func = int(partes[0])
                        nome_func = partes[1]

                        # Listar imagens
                        imagens = []
                        for arquivo in os.listdir(caminho):
                            if arquivo.lower().endswith(('.jpg', '.jpeg', '.png')):
                                img_path = os.path.join(caminho, arquivo)
                                # Ler imagem em escala de cinza
                                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                                if img is not None:
                                    # Redimensionar para tamanho padrão
                                    img = cv2.resize(img, (200, 200))
                                    # Equalizar histograma para melhor contraste
                                    img = cv2.equalizeHist(img)
                                    faces.append(img)
                                    ids.append(id_func)
                                    imagens.append(arquivo)

                        if len(imagens) > 0:
                            funcionarios += 1
                            imagens_por_funcionario[nome_func] = len(imagens)
                            self.log(f"✅ {nome_func}: {len(imagens)} imagens")

                    except ValueError as e:
                        self.log(f"⚠️ Pasta ignorada: {pasta}")
                        continue
                    except Exception as e:
                        self.log(f"⚠️ Erro ao processar {pasta}: {str(e)}")
                        continue

                if len(faces) == 0:
                    self.fila.put(("erro_treinamento", "Nenhuma imagem válida encontrada!"))
                    return

                # Converter para numpy array
                faces_array = np.array(faces, dtype=np.uint8)
                ids_array = np.array(ids, dtype=np.int32)

                self.status_treinamento.config(text=f"⚡ Treinando com {len(faces)} imagens...", fg='#3498db')
                self.log(f"⚡ Iniciando treinamento com {len(faces)} imagens de {funcionarios} funcionários")

                # Criar e treinar o reconhecedor LBPH
                recognizer = cv2.face.LBPHFaceRecognizer_create(
                    radius=1,
                    neighbors=8,
                    grid_x=8,
                    grid_y=8,
                    threshold=80.0
                )

                # Treinar o modelo
                recognizer.train(faces_array, ids_array)

                # Salvar o modelo
                recognizer.write("recognizer/trainer.yml")

                # Atualizar cache de nomes
                self.atualizar_cache_nomes()

                # Estatísticas finais
                total_imagens = len(faces)
                mensagem = f"✅ Treinamento concluído!\n• {total_imagens} imagens\n• {funcionarios} funcionários"

                self.fila.put(("sucesso_treinamento", mensagem))
                self.log(f"✅ Modelo treinado com sucesso! {total_imagens} imagens, {funcionarios} funcionários")

                # Atualizar estatísticas na interface
                self.label_funcionarios.config(text=f"Funcionários cadastrados: {len(self.cache_nomes)}")
                self.label_imagens.config(text=f"Total de imagens: {self.contar_total_imagens()}")

            except Exception as e:
                erro_msg = f"Erro no treinamento: {str(e)}"
                self.fila.put(("erro_treinamento", erro_msg))
                self.log(f"❌ {erro_msg}")

        # Desabilitar botão durante o treinamento
        self.btn_treinar.config(state='disabled', bg='#95a5a6', text="TREINANDO...")

        # Iniciar thread de treinamento
        thread = threading.Thread(target=treinar, daemon=True)
        thread.start()

    def atualizar_cache_nomes(self):
        """Atualiza o cache de nomes"""
        self.cache_nomes.clear()
        if os.path.exists("faces"):
            for pasta in os.listdir("faces"):
                if os.path.isdir(os.path.join("faces", pasta)):
                    try:
                        partes = pasta.split("_", 1)
                        if len(partes) == 2:
                            id_func = int(partes[0])
                            nome = partes[1]
                            self.cache_nomes[id_func] = nome
                    except:
                        continue

        # Atualizar label na aba de treinamento se existir
        if hasattr(self, 'label_funcionarios'):
            self.label_funcionarios.config(text=f"Funcionários cadastrados: {len(self.cache_nomes)}")

        total_imagens = self.contar_total_imagens()
        if hasattr(self, 'label_imagens'):
            self.label_imagens.config(text=f"Total de imagens: {total_imagens}")

    def parar_cadastro(self):
        """Para o cadastro"""
        self.capturando = False
        if hasattr(self, 'btn_iniciar_cadastro'):
            self.btn_iniciar_cadastro.config(state='normal', bg='#27ae60')
        if hasattr(self, 'btn_parar_cadastro'):
            self.btn_parar_cadastro.config(state='disabled', bg='#e74c3c')
        self.log("⏹ Cadastro interrompido")

    def parar_reconhecimento(self):
        """Para o reconhecimento"""
        self.reconhecendo = False
        if hasattr(self, 'btn_iniciar_reconhecimento'):
            self.btn_iniciar_reconhecimento.config(state='normal', bg='#27ae60')
        if hasattr(self, 'btn_parar_reconhecimento'):
            self.btn_parar_reconhecimento.config(state='disabled', bg='#e74c3c')
        self.log("⏹ Reconhecimento interrompido")

    def limpar_registros_hoje(self):
        """Limpa os registros do dia atual"""
        if messagebox.askyesno("Confirmar", "Deseja limpar todos os registros de hoje?"):
            try:
                arquivo = "registros/presenca.csv"
                if os.path.exists(arquivo):
                    df = pd.read_csv(arquivo)
                    hoje = datetime.now().strftime('%Y-%m-%d')
                    # Manter apenas registros de outros dias
                    df_outros = df[df['Data'] != hoje]
                    df_outros.to_csv(arquivo, index=False)

                # Limpar set e treeview
                self.registros_hoje.clear()
                for item in self.tree_registros.get_children():
                    self.tree_registros.delete(item)

                self.log("🗑 Registros de hoje limpos")
                messagebox.showinfo("Sucesso", "Registros de hoje foram limpos!")
            except Exception as e:
                self.log(f"Erro ao limpar registros: {e}")
                messagebox.showerror("Erro", f"Erro ao limpar registros: {e}")

    def registrar_presenca(self, id_func, nome, confianca):
        """Registra presença"""
        try:
            hora = datetime.now()

            # Verificar novamente se já registrou hoje (double-check)
            if id_func in self.registros_hoje:
                self.log(f"⚠️ {nome} já registrado hoje")
                return

            registro = {
                "ID": id_func,
                "Nome": nome,
                "Data": hora.strftime('%Y-%m-%d'),
                "Hora": hora.strftime('%H:%M:%S'),
                "Confianca": f"{confianca:.1f}%"
            }

            arquivo = "registros/presenca.csv"

            # Salvar no CSV
            if not os.path.exists(arquivo):
                pd.DataFrame([registro]).to_csv(arquivo, index=False)
            else:
                df = pd.read_csv(arquivo)
                df = pd.concat([df, pd.DataFrame([registro])], ignore_index=True)
                df.to_csv(arquivo, index=False)

            # Adicionar ao set de registros de hoje
            self.registros_hoje.add(id_func)

            # Atualizar treeview
            if hasattr(self, 'tree_registros'):
                self.tree_registros.insert('', 0, values=(
                    hora.strftime('%H:%M:%S'),
                    nome,
                    id_func,
                    f"{confianca:.1f}%"
                ))

                # Limitar número de registros na treeview
                if len(self.tree_registros.get_children()) > 20:
                    self.tree_registros.delete(self.tree_registros.get_children()[-1])

            self.log(f"✅ REGISTRADO: {nome} (ID: {id_func}) - Confiança: {confianca:.1f}%")

            # Mostrar mensagem visual na interface
            self.status_bar.config(text=f"✅ Registrado: {nome} às {hora.strftime('%H:%M:%S')}")

        except Exception as e:
            self.log(f"❌ Erro ao registrar: {e}")

    def log(self, mensagem):
        """Adiciona mensagem ao log"""
        timestamp = datetime.now().strftime('%H:%M:%S')
        if hasattr(self, 'log_text'):
            self.log_text.insert(tk.END, f"[{timestamp}] {mensagem}\n")
            self.log_text.see(tk.END)
        if hasattr(self, 'status_bar'):
            self.status_bar.config(text=f"{mensagem}")

    def atualizar_interface(self):
        """Atualiza a interface"""
        try:
            while not self.fila.empty():
                tipo, valor = self.fila.get_nowait()

                if tipo == "concluido":
                    messagebox.showinfo("✅ Concluído", valor)
                    self.parar_cadastro()
                    self.atualizar_cache_nomes()
                    # Atualizar contadores
                    if hasattr(self, 'label_funcionarios'):
                        self.label_funcionarios.config(text=f"Funcionários cadastrados: {len(self.cache_nomes)}")
                    if hasattr(self, 'label_imagens'):
                        self.label_imagens.config(text=f"Total de imagens: {self.contar_total_imagens()}")

                elif tipo == "aviso":
                    messagebox.showwarning("⚠️ Aviso", valor)
                    self.parar_cadastro()

                elif tipo == "erro":
                    messagebox.showerror("❌ Erro", valor)
                    self.parar_cadastro()
                    self.parar_reconhecimento()

                elif tipo == "registrar":
                    id_func, nome, conf = valor
                    self.registrar_presenca(id_func, nome, conf)

                elif tipo == "sucesso_treinamento":
                    if hasattr(self, 'status_treinamento'):
                        self.status_treinamento.config(text=valor, fg='#27ae60')
                    if hasattr(self, 'btn_treinar'):
                        self.btn_treinar.config(state='normal', bg='#3498db', text="🎯 TREINAR MODELO")
                    messagebox.showinfo("✅ Sucesso", valor)
                    self.log(valor)
                    self.atualizar_cache_nomes()

                elif tipo == "erro_treinamento":
                    if hasattr(self, 'status_treinamento'):
                        self.status_treinamento.config(text=valor, fg='#e74c3c')
                    if hasattr(self, 'btn_treinar'):
                        self.btn_treinar.config(state='normal', bg='#3498db', text="🎯 TREINAR MODELO")
                    messagebox.showerror("❌ Erro", valor)

        except queue.Empty:
            pass

        self.root.after(100, self.atualizar_interface)

    def on_closing(self):
        """Fecha o sistema"""
        self.capturando = False
        self.reconhecendo = False
        time.sleep(0.5)
        if self.webcam:
            self.webcam.release()
        if self.webcam_cadastro:
            self.webcam_cadastro.release()
        self.root.destroy()


def main():
    root = tk.Tk()
    app = SistemaReconhecimentoFacial(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()


if __name__ == "__main__":
    main()