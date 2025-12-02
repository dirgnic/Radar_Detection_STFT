"""
Modul pentru vizualizarea datelor radar
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional
from matplotlib.patches import Circle
import matplotlib.animation as animation


class RadarVisualizer:
    """
    Clasa pentru vizualizarea datelor și rezultatelor radar
    """
    
    def __init__(self, style: str = 'dark_background'):
        """
        Inițializează visualizer-ul
        
        Args:
            style: Stilul matplotlib
        """
        plt.style.use(style)
        sns.set_palette("husl")
        
    def plot_signals(self,
                    time: np.ndarray,
                    tx_signal: np.ndarray,
                    rx_signal: np.ndarray,
                    if_signal: Optional[np.ndarray] = None,
                    save_path: Optional[str] = None):
        """
        Vizualizează semnalele TX, RX și IF
        
        Args:
            time: Vector timp
            tx_signal: Semnal transmis
            rx_signal: Semnal recepționat
            if_signal: Semnal IF (opțional)
            save_path: Cale pentru salvare
        """
        num_plots = 3 if if_signal is not None else 2
        fig, axes = plt.subplots(num_plots, 1, figsize=(12, 8))
        
        # Afișăm doar primele 1000 de eșantioane pentru claritate
        n_samples = min(1000, len(time))
        
        # Semnal TX
        axes[0].plot(time[:n_samples] * 1e6, np.real(tx_signal[:n_samples]), 
                    'c-', linewidth=0.5, label='Real')
        axes[0].set_ylabel('Amplitudine')
        axes[0].set_title('Semnal Transmis (TX)', fontsize=12, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        
        # Semnal RX
        axes[1].plot(time[:n_samples] * 1e6, np.real(rx_signal[:n_samples]), 
                    'g-', linewidth=0.5, label='Real')
        axes[1].set_ylabel('Amplitudine')
        axes[1].set_title('Semnal Recepționat (RX)', fontsize=12, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
        
        # Semnal IF
        if if_signal is not None:
            axes[2].plot(time[:n_samples] * 1e6, np.real(if_signal[:n_samples]), 
                        'y-', linewidth=0.5, label='Real')
            axes[2].set_xlabel('Timp (μs)')
            axes[2].set_ylabel('Amplitudine')
            axes[2].set_title('Semnal IF (După Mixer)', fontsize=12, fontweight='bold')
            axes[2].grid(True, alpha=0.3)
            axes[2].legend()
        else:
            axes[1].set_xlabel('Timp (μs)')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_spectrum(self,
                     freqs: np.ndarray,
                     spectrum: np.ndarray,
                     detected_peaks: Optional[np.ndarray] = None,
                     title: str = 'Spectru FFT',
                     save_path: Optional[str] = None):
        """
        Vizualizează spectrul de frecvență
        
        Args:
            freqs: Vector frecvențe
            spectrum: Magnitudine spectru (dB)
            detected_peaks: Indicii vârfurilor detectate
            title: Titlul graficului
            save_path: Cale pentru salvare
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Spectru
        ax.plot(freqs / 1e3, spectrum, 'c-', linewidth=1, label='Spectru')
        
        # Marcăm vârfurile detectate
        if detected_peaks is not None and len(detected_peaks) > 0:
            ax.plot(freqs[detected_peaks] / 1e3, spectrum[detected_peaks], 
                   'r*', markersize=15, label=f'Ținte detectate ({len(detected_peaks)})')
            
            # Adăugăm adnotări
            for peak in detected_peaks:
                ax.annotate(f'{freqs[peak]/1e3:.1f} kHz',
                          xy=(freqs[peak]/1e3, spectrum[peak]),
                          xytext=(10, 10), textcoords='offset points',
                          bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                          arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        ax.set_xlabel('Frecvență (kHz)', fontsize=11)
        ax.set_ylabel('Magnitudine (dB)', fontsize=11)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_range_doppler(self,
                          range_axis: np.ndarray,
                          doppler_axis: np.ndarray,
                          rd_map: np.ndarray,
                          save_path: Optional[str] = None):
        """
        Vizualizează harta distanță-Doppler
        
        Args:
            range_axis: Axă distanță
            doppler_axis: Axă Doppler
            rd_map: Matrice 2D distanță-Doppler
            save_path: Cale pentru salvare
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Conversie axe în unități fizice
        range_m = range_axis
        velocity_mps = doppler_axis
        
        # Plot
        im = ax.imshow(rd_map, 
                      aspect='auto',
                      origin='lower',
                      extent=[range_m[0], range_m[-1], 
                             velocity_mps[0], velocity_mps[-1]],
                      cmap='jet',
                      interpolation='bilinear')
        
        ax.set_xlabel('Distanță (m)', fontsize=11)
        ax.set_ylabel('Viteză (m/s)', fontsize=11)
        ax.set_title('Hartă Distanță-Doppler', fontsize=13, fontweight='bold')
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Magnitudine (dB)', fontsize=10)
        
        # Linii de referință
        ax.axhline(y=0, color='w', linestyle='--', alpha=0.5, linewidth=1)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_ppi(self,
                targets: List,
                max_range: float,
                save_path: Optional[str] = None):
        """
        Vizualizează PPI (Plan Position Indicator) - radar view
        
        Args:
            targets: Lista de ținte detectate
            max_range: Raza maximă de afișare
            save_path: Cale pentru salvare
        """
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        # Configurare axe polare
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)
        ax.set_ylim(0, max_range)
        
        # Cercuri concentrice
        for r in np.linspace(0, max_range, 5):
            circle = Circle((0, 0), r, fill=False, color='green', 
                          alpha=0.3, linewidth=1, transform=ax.transData)
        
        # Grilă radială
        ax.grid(True, color='green', alpha=0.3)
        
        # Plot ținte
        for target in targets:
            # Presupunem unghiul uniform distribuit pentru demonstrație
            angle = np.random.uniform(0, 2*np.pi)
            
            # Marker
            ax.plot(angle, target.range, 'ro', markersize=10, 
                   markerfacecolor='red', markeredgecolor='yellow', 
                   markeredgewidth=2)
            
            # Adnotare
            ax.text(angle, target.range, 
                   f'\n{target.range/1000:.1f} km\n{target.velocity:.0f} m/s',
                   fontsize=8, ha='center',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))
        
        ax.set_title('Plan Position Indicator (PPI)', 
                    fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_spectrogram(self,
                        times: np.ndarray,
                        freqs: np.ndarray,
                        spectrogram: np.ndarray,
                        save_path: Optional[str] = None):
        """
        Vizualizează spectrograma
        
        Args:
            times: Vector timp
            freqs: Vector frecvențe
            spectrogram: Matrice spectrogramă
            save_path: Cale pentru salvare
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        im = ax.pcolormesh(times * 1e3, freqs / 1e3, spectrogram, 
                          shading='gouraud', cmap='viridis')
        
        ax.set_xlabel('Timp (ms)', fontsize=11)
        ax.set_ylabel('Frecvență (kHz)', fontsize=11)
        ax.set_title('Spectrogramă - Analiză Timp-Frecvență', 
                    fontsize=13, fontweight='bold')
        
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Magnitudine (dB)', fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_target_summary(self,
                          targets: List,
                          save_path: Optional[str] = None):
        """
        Vizualizează sumar ținte detectate
        
        Args:
            targets: Lista de ținte
            save_path: Cale pentru salvare
        """
        if not targets:
            print("Nu există ținte de vizualizat")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        ranges = [t.range / 1000 for t in targets]  # km
        velocities = [t.velocity for t in targets]  # m/s
        snrs = [t.snr for t in targets]  # dB
        
        # Histogramă distanțe
        axes[0, 0].bar(range(len(ranges)), ranges, color='cyan', edgecolor='white')
        axes[0, 0].set_xlabel('Index țintă')
        axes[0, 0].set_ylabel('Distanță (km)')
        axes[0, 0].set_title('Distanțe ținte detectate')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Histogramă viteze
        axes[0, 1].bar(range(len(velocities)), velocities, color='green', edgecolor='white')
        axes[0, 1].set_xlabel('Index țintă')
        axes[0, 1].set_ylabel('Viteză (m/s)')
        axes[0, 1].set_title('Viteze ținte detectate')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
        
        # Scatter distanță vs viteză
        axes[1, 0].scatter(ranges, velocities, s=100, c=snrs, 
                          cmap='hot', edgecolors='white', linewidth=1)
        axes[1, 0].set_xlabel('Distanță (km)')
        axes[1, 0].set_ylabel('Viteză (m/s)')
        axes[1, 0].set_title('Distanță vs Viteză')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].axhline(y=0, color='cyan', linestyle='--', alpha=0.5)
        
        # Histogramă SNR
        axes[1, 1].bar(range(len(snrs)), snrs, color='yellow', edgecolor='white')
        axes[1, 1].set_xlabel('Index țintă')
        axes[1, 1].set_ylabel('SNR (dB)')
        axes[1, 1].set_title('Raport Semnal-Zgomot')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle(f'Sumar {len(targets)} Ținte Detectate', 
                    fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
