import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


DISEASE_RATES = {
    'Dispozitiv de suport':         49.4,
    'Opacitate pulmonară':          34.6,
    'Leziune pulmonară':            10.4,
    'Atelectazie':                  10.1,
    'Consolidare':                   9.1,
    'Efuziune pleurală':             9.0,
    'Pleurezie':                     4.2,
    'Lărgire cardiomediastinală':    3.5,
    'Pneumotorace':                  3.4,
    'Cardiomegalie':                 3.2,
    'Fractură':                      2.0,
    'Pneumonie':                     1.8,
    'Edem pulmonar':                 1.3,
}


diseases = list(DISEASE_RATES.keys())
rates    = list(DISEASE_RATES.values())

fig, ax = plt.subplots(figsize=(11, 7))


colors = ['#e07856' if r < 10 else '#4f7ef8' for r in rates]

bars = ax.barh(diseases, rates, color=colors, alpha=0.85, edgecolor='white')


for bar, rate in zip(bars, rates):
    ax.text(rate + 0.5, bar.get_y() + bar.get_height()/2,
            f'{rate}%', va='center', fontsize=10, fontweight='bold')

ax.set_xlabel('Procentul de imagini în care apare patologia (%)', fontsize=11)
ax.set_title('Distribuția patologiilor în setul de antrenament\n(78.506 imagini)',
             fontsize=13, fontweight='bold', pad=15)
ax.set_xlim(0, 55)
ax.invert_yaxis()  
ax.grid(axis='x', alpha=0.3)


from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#4f7ef8', alpha=0.85, label='Clase bine reprezentate (≥ 10%)'),
    Patch(facecolor='#e07856', alpha=0.85, label='Clase subreprezentate (< 10%)'),
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=10)

plt.tight_layout()
plt.savefig(r'C:\xray-project\results\class_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved class_distribution.png')