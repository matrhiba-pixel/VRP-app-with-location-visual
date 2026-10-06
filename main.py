# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 18:54:18 2026

@author: Aymane
"""

import csv
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.optimize import milp, LinearConstraint, Bounds

class PCenterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Optimisation p-Centre - Localisation de Sites")
        self.root.geometry("1200x700")

        # Données
        self.clients = []
        self.sites = []
        self.solution = None

        self._build_ui()
        self._load_default_sample_data()

    def _build_ui(self):
        # Frame Principal
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # --- PANNEAU GAUCHE (Boutons & Paramètres) ---
        left_frame = ttk.LabelFrame(main_frame, text=" Contrôles ", width=300)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        left_frame.pack_propagate(False)

        # Paramètre p
        ttk.Label(left_frame, text="Nombre max de sites à ouvrir (p) :", font=("Arial", 10, "bold")).pack(anchor=tk.W, padx=10, pady=(15, 2))
        self.entry_p = ttk.Entry(left_frame, font=("Arial", 10))
        self.entry_p.insert(0, "2")
        self.entry_p.pack(fill=tk.X, padx=10, pady=(0, 15))

        # Bouton 1 : Read File
        btn_read = ttk.Button(left_frame, text="1. Read File (Charger Fichiers)", command=self.load_files)
        btn_read.pack(fill=tk.X, padx=10, pady=8)

        # Bouton 2 : Solve Problem
        btn_solve = ttk.Button(left_frame, text="2. Solve Problem (Résoudre)", command=self.solve_problem)
        btn_solve.pack(fill=tk.X, padx=10, pady=8)

        # Bouton 3 : Close Application
        btn_close = ttk.Button(left_frame, text="3. Close Application (Fermer)", command=self.root.quit)
        btn_close.pack(fill=tk.X, padx=10, pady=8)

        ttk.Separator(left_frame, orient='horizontal').pack(fill='x', padx=10, pady=15)

        # Informations / Status
        ttk.Label(left_frame, text="Statut du Chargement :", font=("Arial", 9, "bold")).pack(anchor=tk.W, padx=10)
        self.lbl_status = ttk.Label(left_frame, text="Données par défaut chargées.", wraplength=260, foreground="blue")
        self.lbl_status.pack(anchor=tk.W, padx=10, pady=5)

        # --- PANNEAU DROIT (Affichage & Visualisation) ---
        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Layout Droit : Haut = Graphique, Bas = Résultats Textuels
        self.fig, self.ax = plt.subplots(figsize=(7, 5))
        self.canvas = FigureCanvasTkAgg(self.fig, master=right_frame)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Text Box pour la Solution
        text_frame = ttk.LabelFrame(right_frame, text=" Résultats & Raccordements Client-Site ")
        text_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=False, pady=5)

        self.txt_output = tk.Text(text_frame, height=8, font=("Consolas", 9))
        scrollbar = ttk.Scrollbar(text_frame, command=self.txt_output.yview)
        self.txt_output.configure(yscrollcommand=scrollbar.set)
        
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.txt_output.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    def _load_default_sample_data(self):
        """Génère un jeu de données de démonstration au démarrage."""
        self.clients = [
            {'id': f'C{i+1}', 'x': x, 'y': y}
            for i, (x, y) in enumerate([
                (12, 24), (34, 81), (78, 12), (55, 60), 
                (89, 45), (22, 15), (67, 88), (41, 33)
            ])
        ]
        self.sites = [
            {'id': f'S{j+1}', 'x': x, 'y': y}
            for j, (x, y) in enumerate([
                (15, 20), (75, 15), (50, 65), (85, 85)
            ])
        ]
        self.plot_map()

    def load_files(self):
        """Lit les fichiers CSV pour les clients et les sites."""
        c_path = filedialog.askopenfilename(title="Sélectionner le fichier CSV des Clients (id, x, y)")
        if not c_path:
            return
        
        s_path = filedialog.askopenfilename(title="Sélectionner le fichier CSV des Sites (id, x, y)")
        if not s_path:
            return

        try:
            clients, sites = [], []
            with open(c_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for r in reader:
                    clients.append({'id': r['id'], 'x': float(r['x']), 'y': float(r['y'])})

            with open(s_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for r in reader:
                    sites.append({'id': r['id'], 'x': float(r['x']), 'y': float(r['y'])})

            self.clients = clients
            self.sites = sites
            self.solution = None
            self.lbl_status.config(text=f"Chargé : {len(clients)} clients, {len(sites)} sites.", foreground="green")
            self.plot_map()
            messagebox.showinfo("Succès", "Fichiers chargés avec succès !")
        except Exception as e:
            messagebox.showerror("Erreur", f"Échec du chargement des fichiers :\n{str(e)}")

    def solve_problem(self):
        """Résout le problème du p-Centre sous contraintes ILP."""
        if not self.clients or not self.sites:
            messagebox.showwarning("Attention", "Veuillez d'abord charger les données.")
            return

        try:
            p = int(self.entry_p.get())
            if p <= 0 or p > len(self.sites):
                raise ValueError(f"p doit être compris entre 1 et {len(self.sites)}")
        except ValueError as ve:
            messagebox.showerror("Paramètre invalide", f"Veuillez entrer une valeur valide pour p :\n{str(ve)}")
            return

        N = len(self.clients)
        M = len(self.sites)

        c_coords = np.array([[c['x'], c['y']] for c in self.clients])
        s_coords = np.array([[s['x'], s['y']] for s in self.sites])

        # Matrice des distances euclidiennes
        dist = np.linalg.norm(c_coords[:, np.newaxis, :] - s_coords[np.newaxis, :, :], axis=2)

        # Variables: [y_0..y_M-1, x_00..x_NM-1, D_max]
        num_vars = M + N * M + 1
        c_obj = np.zeros(num_vars)
        c_obj[-1] = 1.0  # Minimiser D_max

        integrality = np.ones(num_vars)
        integrality[-1] = 0  # D_max est continu

        bounds = Bounds(lb=np.zeros(num_vars), ub=np.array([1.0]*M + [1.0]*(N*M) + [np.inf]))
        constraints = []

        # 1. Un client i rattaché à un seul site: sum_j x_ij = 1
        A_assign = np.zeros((N, num_vars))
        for i in range(N):
            for j in range(M):
                A_assign[i, M + i * M + j] = 1.0
        constraints.append(LinearConstraint(A_assign, lb=np.ones(N), ub=np.ones(N)))

        # 2. Site ouvert: x_ij - y_j <= 0
        A_open = np.zeros((N * M, num_vars))
        row = 0
        for i in range(N):
            for j in range(M):
                A_open[row, j] = -1.0
                A_open[row, M + i * M + j] = 1.0
                row += 1
        constraints.append(LinearConstraint(A_open, lb=-np.inf * np.ones(N * M), ub=np.zeros(N * M)))

        # 3. Nombre max de sites ouverts: sum_j y_j <= p
        A_p = np.zeros((1, num_vars))
        A_p[0, :M] = 1.0
        constraints.append(LinearConstraint(A_p, lb=0, ub=p))

        # 4. Distance max: sum_j d_ij * x_ij - D_max <= 0
        A_dist = np.zeros((N, num_vars))
        for i in range(N):
            for j in range(M):
                A_dist[i, M + i * M + j] = dist[i, j]
            A_dist[i, -1] = -1.0
        constraints.append(LinearConstraint(A_dist, lb=-np.inf * np.ones(N), ub=np.zeros(N)))

        # Résolution
        res = milp(c=c_obj, integrality=integrality, bounds=bounds, constraints=constraints)

        if res.success:
            sol = res.x
            open_sites_idx = np.where(sol[:M] > 0.5)[0]
            assignments = []
            for i in range(N):
                for j in range(M):
                    if sol[M + i * M + j] > 0.5:
                        assignments.append({
                            'client_id': self.clients[i]['id'],
                            'client_coord': (self.clients[i]['x'], self.clients[i]['y']),
                            'site_id': self.sites[j]['id'],
                            'site_coord': (self.sites[j]['x'], self.sites[j]['y']),
                            'dist': dist[i, j]
                        })
            d_max = sol[-1]
            self.solution = {
                'open_sites': [self.sites[j] for j in open_sites_idx],
                'assignments': assignments,
                'd_max': d_max
            }
            self.display_results()
            self.plot_map()
        else:
            messagebox.showerror("Résolution impossible", "Impossible de trouver une solution optimale.")

    def display_results(self):
        """Affiche les résultats dans la zone de texte."""
        self.txt_output.delete('1.0', tk.END)
        if not self.solution:
            return

        open_ids = [s['id'] for s in self.solution['open_sites']]
        out = []
        out.append("=== RÉSULTATS DE L'OPTIMISATION ===")
        out.append(f"Distance Maximale Optimale (D_max) : {self.solution['d_max']:.2f}")
        out.append(f"Sites Ouverts ({len(open_ids)}) : {', '.join(open_ids)}")
        out.append("-" * 55)
        out.append(f"{'Client ID':<12} | {'Site Rattaché':<15} | {'Distance':<10}")
        out.append("-" * 55)
        for a in self.solution['assignments']:
            out.append(f"{a['client_id']:<12} | {a['site_id']:<15} | {a['dist']:.2f}")

        self.txt_output.insert(tk.END, "\n".join(out))

    def plot_map(self):
        """Affiche la carte des emplacements et le raccordement client-site."""
        self.ax.clear()

        # Coordonnées Clients
        cx = [c['x'] for c in self.clients]
        cy = [c['y'] for c in self.clients]
        self.ax.scatter(cx, cy, c='blue', marker='o', label='Clients', s=50, zorder=3)
        for c in self.clients:
            self.ax.annotate(c['id'], (c['x'], c['y']), textcoords="offset points", xytext=(5, 5), fontsize=8)

        # Coordonnées Sites
        sx = [s['x'] for s in self.sites]
        sy = [s['y'] for s in self.sites]
        
        if self.solution:
            open_ids = {s['id'] for s in self.solution['open_sites']}
            closed_x = [s['x'] for s in self.sites if s['id'] not in open_ids]
            closed_y = [s['y'] for s in self.sites if s['id'] not in open_ids]
            open_x = [s['x'] for s in self.sites if s['id'] in open_ids]
            open_y = [s['y'] for s in self.sites if s['id'] in open_ids]

            if closed_x:
                self.ax.scatter(closed_x, closed_y, c='gray', marker='^', label='Sites Fermés', s=70, alpha=0.5, zorder=2)
            if open_x:
                self.ax.scatter(open_x, open_y, c='red', marker='^', label='Sites Ouverts', s=120, zorder=4)

            # Lignes d'affectation
            for a in self.solution['assignments']:
                c_x, c_y = a['client_coord']
                s_x, s_y = a['site_coord']
                self.ax.plot([c_x, s_x], [c_y, s_y], 'g--', alpha=0.6, linewidth=1)
        else:
            self.ax.scatter(sx, sy, c='red', marker='^', label='Sites Potentiels', s=80, zorder=3)

        for s in self.sites:
            self.ax.annotate(s['id'], (s['x'], s['y']), textcoords="offset points", xytext=(5, 5), fontsize=8, fontweight='bold')

        self.ax.set_title("Carte de Localisation & Affectations Clients - Sites")
        self.ax.set_xlabel("Coordonnée X")
        self.ax.set_ylabel("Coordonnée Y")
        self.ax.legend(loc='upper right')
        self.ax.grid(True, linestyle=':', alpha=0.6)
        self.fig.tight_layout()
        self.canvas.draw()

if __name__ == "__main__":
    root = tk.Tk()
    app = PCenterApp(root)
    root.mainloop()