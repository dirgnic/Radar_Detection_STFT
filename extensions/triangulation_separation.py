import os
import numpy as np
from typing import List, Tuple, Set
from dataclasses import dataclass
from scipy import signal
import matplotlib.pyplot as plt

def maximum_filter_2d(image, size = 3):         # filtru pentru punctele de intensitate maxima
    h, w = image.shape
    result = np.zeros_like(image)
    half = size // 2
    for i in range(h):
        for j in range(w):
            i_min = max(0, i - half)
            i_max = min(h, i + half + 1)
            j_min = max(0, j - half)
            j_max = min(w, j + half + 1)
            window = image[i_min:i_max, j_min:j_max]                # salveaza valoarea maxima
            result[i, j] = np.max(window)
    return result


def find_local_maxima(image, window_size = 3):              # identifica punctele care sunt maxime locale
    max_filtered = maximum_filter_2d(image, window_size)
    local_max = (image == max_filtered)
    return local_max

@dataclass
class Triangle:
    vertices: Tuple[int, int, int]
    def __hash__(self):
        return hash(tuple(sorted(self.vertices)))
    def __eq__(self, other):
        return set(self.vertices) == set(other.vertices)
    def contains_vertex(self, v: int) -> bool:
        return v in self.vertices
    def get_edge_opposite(self, v: int) -> Tuple[int, int]:
        verts = list(self.vertices)
        verts.remove(v)
        return tuple(sorted(verts))                 # gaseste latura opusa unui varf, prin eliminarea lui

class DelaunayTriangulation:
    def __init__(self, points):
        self.points = points.astype(float)
        self.n_points = len(points)
        self.triangles: List[Triangle] = []
        self.triangulate()
    def triangulate(self):
        super_tri = self.create_super_triangle()
        self.triangles = [super_tri]
        for i in range(self.n_points):
            self.add_point(i)
        super_vertices = {self.n_points, self.n_points + 1, self.n_points + 2}
        self.triangles = [tri for tri in self.triangles if not any(v in super_vertices for v in tri.vertices)]

    def create_super_triangle(self):            # creeaza triunghiul ce cuprinde toate punctele
        min_x, min_y = np.min(self.points, axis=0)
        max_x, max_y = np.max(self.points, axis=0)
        dx = max_x - min_x
        dy = max_y - min_y
        delta_max = max(dx, dy) * 10                        # extinde cadrul in exterior
        mid_x = (min_x + max_x) / 2                     # mijlocul pe fiecare axa
        mid_y = (min_y + max_y) / 2
        p1 = np.array([mid_x - delta_max, mid_y - delta_max])           # varfurile triunghiului mare
        p2 = np.array([mid_x + delta_max, mid_y - delta_max])
        p3 = np.array([mid_x, mid_y + delta_max])
        self.points = np.vstack([self.points, p1, p2, p3])
        return Triangle((self.n_points, self.n_points + 1, self.n_points + 2))

    def add_point(self, point_idx):
        point = self.points[point_idx]          # adauga un punct si triunghiurile aferente
        bad_triangles = []
        for tri in self.triangles:
            if self.in_circumcircle(point, tri):            # daca punctul e in interiorul cercului circumscris
                bad_triangles.append(tri)               # oricarui triunghi, atunci nu e bun
        polygon = []
        for tri in bad_triangles:           # gaseste laturile care nu sunt comune in triunghiurile rele
            for i in range(3):
                edge = self.get_edge(tri, i)        # daca latura i e comuna, e in interiorul poligonului
                is_shared = False
                for other_tri in bad_triangles:
                    if tri == other_tri:
                        continue
                    if self.triangle_has_edge(other_tri, edge):
                        is_shared = True
                        break
                if not is_shared:
                    polygon.append(edge)
        for tri in bad_triangles:
            self.triangles.remove(tri)
        for edge in polygon:                        # conecteaza punctul nou cu fiecare latura a poligonului
            new_tri = Triangle((point_idx, edge[0], edge[1]))           # formeaza triunghiuri noi
            self.triangles.append(new_tri)

    def in_circumcircle(self, point, tri):              # verificarea daca e in cercul circumscris
        a = self.points[tri.vertices[0]]
        b = self.points[tri.vertices[1]]
        c = self.points[tri.vertices[2]]                # prin calculul determinantului (daca e mai mare ca 0, e in cerc)
        a = a - point
        b = b - point
        c = c - point
        det = np.linalg.det([[a[0], a[1], a[0]**2 + a[1]**2], [b[0], b[1], b[0]**2 + b[1]**2], [c[0], c[1], c[0]**2 + c[1]**2]])
        return det > 0

    def get_edge(self, tri, edge_idx):
        v = list(tri.vertices)                  # returneaza o latura a triunghiului
        if edge_idx == 0:
            return (v[0], v[1])
        elif edge_idx == 1:
            return (v[1], v[2])
        else:
            return (v[2], v[0])

    def triangle_has_edge(self, tri, edge):         # verifica daca o muchie apartine unui triunghi
        edges = [self.get_edge(tri, 0), self.get_edge(tri, 1),self.get_edge(tri, 2)]
        edge_set = set(edge)
        for e in edges:
            if set(e) == edge_set:
                return True
        return False

    def simplices(self):            # obtine triunghiurile ca matrice
        return np.array([list(tri.vertices) for tri in self.triangles])

@dataclass
class TriangulatedComponent:
    component_id: int               # id-ul componentei
    points: np.ndarray
    triangles: np.ndarray
    energy: float                   # energia totala
    centroid_freq: float
    centroid_time: float

class TriangulationSeparator:               # separatorul prin triangulare
    def __init__(self, peak_threshold_percentile: float = 90, min_peak_distance: int = 3, connectivity_threshold: float = 0.1):
        self.peak_threshold_percentile = peak_threshold_percentile
        self.min_peak_distance = min_peak_distance
        self.connectivity_threshold = connectivity_threshold
        self.stft_magnitude = None
        self.freqs = None
        self.times = None
        self.peaks = None
        self.triangulation = None
        self.components = []

    def detect_peaks(self, magnitude):          # varfurile
        threshold = np.percentile(magnitude, self.peak_threshold_percentile)        # prag adaptiv
        local_max = find_local_maxima(magnitude, self.min_peak_distance)        # gaseste maximele locale
        peaks = local_max & (magnitude > threshold)         # si le retine pe toate peste prag
        peak_coords = np.array(np.where(peaks)).T       # converteste peakurile in lista de coordonate
        print(f"Au fost detectate {len(peak_coords)} varfuri")
        self.peaks = peak_coords
        return peak_coords

    def triangulate(self):              # creeaza triangularea
        peak_coords_real = np.zeros_like(self.peaks, dtype=float)
        freq_indices = self.peaks[:, 0]
        time_indices = self.peaks[:, 1]
        peak_coords_real[:, 0] = self.freqs[freq_indices] / 100.0
        peak_coords_real[:, 1] = self.times[time_indices] / 0.05
        tri = DelaunayTriangulation(peak_coords_real)
        print(f"S-au creat {len(tri.simplices)} triunghiuri")
        self.triangulation = tri
        return tri

    def group_triangles(self):
        tri = self.triangulation
        magnitude = self.stft_magnitude
        simplices = tri.simplices
        n_triangles = len(simplices)
        triangle_labels = np.full(n_triangles, -1, dtype=int)
        edge_to_triangles = {}
        for i, s in enumerate(simplices):
            edges = [tuple(sorted((s[0], s[1]))), tuple(sorted((s[1], s[2]))),tuple(sorted((s[2], s[0])))]
            for e in edges:         # marcheaza triunghiurile care au laturi comune
                if e not in edge_to_triangles:
                    edge_to_triangles[e] = []
                edge_to_triangles[e].append(i)
        adjacency = {i: [] for i in range(n_triangles)}
        for sharing_triangles in edge_to_triangles.values():
            if len(sharing_triangles) == 2:         # daca latura e in exact 2 triunghiuri
                i, j = sharing_triangles
                v_i = self.peaks[simplices[i]]
                v_j = self.peaks[simplices[j]]
                mag_i = np.mean([magnitude[v[0], v[1]] for v in v_i])
                mag_j = np.mean([magnitude[v[0], v[1]] for v in v_j])
                if abs(mag_i - mag_j) / max(mag_i, mag_j) < self.connectivity_threshold:        # atunci vezi daca diferenta relativa a energiei e sub prag
                    adjacency[i].append(j)              # caz in care se adauga legatura la graf
                    adjacency[j].append(i)
        component_id = 0
        for i in range(n_triangles):
            if triangle_labels[i] != -1:
                continue
            stack = [i]
            triangle_labels[i] = component_id           # DFS pentru componentele conexe
            while stack:
                curr = stack.pop()
                for neighbor in adjacency[curr]:
                    if triangle_labels[neighbor] == -1:
                        triangle_labels[neighbor] = component_id
                        stack.append(neighbor)
            component_id += 1
        print(f"Au fost gasite {component_id} componente separate.")
        components = []
        for c_id in range(component_id):
            comp_tri_idx = np.where(triangle_labels == c_id)[0]
            if len(comp_tri_idx) == 0: continue
            pt_indices = np.unique(simplices[comp_tri_idx])     # punctele unice din componenta
            points = self.peaks[pt_indices]
            f_idx, t_idx = points[:, 0], points[:, 1]
            energy = np.sum(magnitude[f_idx, t_idx] ** 2)       # calculul energiei totale
            comp = TriangulatedComponent(component_id=c_id, points=points,
                triangles=simplices[comp_tri_idx], energy=energy, centroid_freq=np.mean(self.freqs[f_idx]), centroid_time=np.mean(self.times[t_idx]))
            components.append(comp)
        components.sort(key=lambda x: x.energy, reverse=True)       # sorteaza descrescator dupa energie
        self.components = components
        return components

    def separate(self, stft_magnitude, freqs, times):
        print("\n[SEPARARE CU TRIANGULARE]")
        self.stft_magnitude = stft_magnitude
        self.freqs = freqs                  # STFT
        self.times = times
        print("Detectia varfurilor")
        self.detect_peaks(stft_magnitude)       # varfuri
        print("Triangulare Delaunay")
        self.triangulate()                  # triangulare
        print("Gruparea componentelor")
        components = self.group_triangles()     # gruparea componentelor dupa energie
        print(f"\nTotal de {len(components)} componente separate")
        return components


def visualize_triangulation(separator, output_path="./spectrograms/triangulation_separation.pdf"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 8))
    sxx_db = 20 * np.log10(separator.stft_magnitude + 1e-6)             # pentru vizualizarea exemplului de mai jos
    im = ax.pcolormesh(separator.times, separator.freqs, sxx_db, shading='gouraud', cmap='magma', alpha=0.7)
    cmap_comp = plt.get_cmap('tab20')
    for i, comp in enumerate(separator.components):
        color = cmap_comp(i % 20)
        y_coords = separator.freqs[separator.peaks[:, 0]]
        x_coords = separator.times[separator.peaks[:, 1]]
        ax.triplot(x_coords, y_coords, comp.triangles, color=color, linewidth=0.5, alpha=0.8)
        if len(comp.points) > 20:
            ax.text(comp.centroid_time, comp.centroid_freq, f"ID {comp.component_id}",
                    color='white', fontsize=8, fontweight='bold', bbox=dict(facecolor=color, alpha=0.6, edgecolor='none'))
    ax.set_title(f"Separare prin Triangulare Delaunay\n{len(separator.components)} Componente Detectate (Fragmentate)")
    ax.set_xlabel("Timp [s]")
    ax.set_ylabel("Frecventa [Hz]")
    plt.colorbar(im, ax=ax, label="Intensitate [dB]")
    plt.savefig(output_path, format='pdf', bbox_inches='tight')
    print(f"Vizualizarea triangularii salvata in: {output_path}")
    plt.show()

def demo_implementation():
    fs = 44100
    duration = 2.0
    t = np.linspace(0, duration, int(fs * duration))
    chirp = signal.chirp(t, 500, duration, 1500) * 0.8
    tone = np.sin(2 * np.pi * 800 * t) * 0.6
    noise = np.random.randn(len(t)) * 0.05
    test_signal = chirp + tone + noise
    window_size = 512
    hop_size = 128
    sigma = window_size / 6
    window = signal.windows.gaussian(window_size, sigma)
    freqs, times, Zxx = signal.stft( test_signal, fs=fs, window=window, nperseg=window_size, noverlap=window_size - hop_size, return_onesided=True)
    magnitude = np.abs(Zxx)
    print(f"STFT shape: {magnitude.shape}")
    separator = TriangulationSeparator(peak_threshold_percentile=92, min_peak_distance=5, connectivity_threshold=0.2)
    components = separator.separate(magnitude, freqs, times)
    print("\nComponente gasite:")
    for comp in components:
        print(f"  Componenta {comp.component_id}: {comp.centroid_freq:.0f} Hz, "
              f"{len(comp.points)} puncte, energie={comp.energy:.2e}")
    visualize_triangulation(separator)

if __name__ == "__main__":
    demo_implementation()