import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import pickle
import numpy as np
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression, Lasso, Ridge, LassoCV, RidgeCV
from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

import funct as funky

# file = open('variables/columns','rb')
# columns = pickle.load(file)
# file.close()
# file = open('variables/columns_list','rb')
# columns_list = pickle.load(file)
# file.close()
file = open('variables/feature_means','rb')
feature_means = np.load(file)
file.close()
file = open('variables/feature_stds','rb')
feature_stds = np.load(file)
file.close()

HZ_FONT = 'Helvetica 12'
HZ_SIZE = (5,1)
DB_SIZE = (3,1)
AC_FREQS = [0.125, 0.25,0.5,1.0,1.5,2.0,3.0,4.0,6.0,8.0]
BC_FREQS = [0.5,1.0,1.5,2.0,3.0,4.0]
FREQS = AC_FREQS + BC_FREQS
X_BONE = np.array(BC_FREQS)
X_AIR = np.array(AC_FREQS)
Y_GRID = np.arange(-10,130,10)
X_GRID = np.array([0.125, 0.25,0.5,0.75,1,1.5,2,3,4,6,8])


CHAR_LIMIT = 3

class Root(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title('Tkinter Validation Demo')

        self.title("Audiogram prediction (after otosclerosis surgery)")
        self.geometry("1200x900")

    def numeric_check(self, P):
        return (str.isdigit(P) or P == "") and len(P) <= CHAR_LIMIT
    
    def gender_check(self, P):
        return (P == "M" or P == "Z" or P == "") and len(P) <= 1
    
    def side_check(self, P):
        return (P == "L" or P == "D" or P == "") and len(P) <= 1

    def add_entry(self, entries, label, key,row,col=0):
        ttk.Label(self, text=label, font=HZ_FONT).grid(row=row,column=col,pady=4)
        if label in ["Spol (Z/M)","Stran glave (L/D)"]:
            if label == "Spol (Z/M)":
                vcmd = (self.register(self.gender_check), '%P')    
            if label == "Stran glave (L/D)":
                vcmd = (self.register(self.side_check), '%P')            
        else:
            vcmd = (self.register(self.numeric_check), '%P')
        e = ttk.Entry(self, validate="all", validatecommand=vcmd,width=6, font=HZ_FONT)
        e.grid(row = row, column = col+1, pady = 2)
        entries[key] = e
        return entries
    
def fig_maker(observed, predicted, x_grid, y_grid, x_air, x_bone):
    plt.close('all')

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xscale('log')
    ax.set_xticks(x_grid)
    ax.set_yticks(y_grid)
    ax.grid(True, which='major')

    ax.set_ylim(-10, 120)
    ax.invert_yaxis()
    ax.set_box_aspect(1)

    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Hearing Level (dB)')

    ax.plot(x_air, observed[:10], color='blue', marker='x')
    ax.plot(x_bone, observed[10:], color='blue', marker='o')
    ax.plot(x_air, predicted[:10], color='orange', marker='x', linestyle='dashed')
    ax.plot(x_bone, predicted[10:], color='orange', marker='o', linestyle='dashed')

    return fig


def draw_figure(container, figure):
    canvas = FigureCanvasTkAgg(figure, master=container)
    canvas.draw()
    canvas.get_tk_widget().grid(container,row=0,col=4,rowspan=2,columnspan=2,padx=2,pady=2)
    return canvas


def add_pred(parent, label, key,row,col=0):
    ttk.Label(parent, text=label, font=HZ_FONT).grid(row=row,column=col,pady=4)
    l = ttk.Label(parent, text="__", font=HZ_FONT, foreground="orange")
    l.grid(row = row, column = col+1, pady = 2)
    pred_labels[key] = l

def validate_entries(entries):
    return

root = Root()
# main = ttk.Frame(root, padding=10)
# main.pack(fill=tk.BOTH, expand=True)

# root = ttk.Frame(main)
# root.pack(side=tk.LEFT, fill=tk.Y)

# right = ttk.Frame(main)
# right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

HZ_FONT = ("Helvetica", 12)

entries = {}

entries = root.add_entry(entries,"Starost","Age",row=0,col=0)
entries = root.add_entry(entries,"Spol (Z/M)","Gender-0=M, 1 =F",row=0,col=2)
entries = root.add_entry(entries,"Stran glave (L/D)","Side 0=L, 1=R",row=1,col=0)


ttk.Label(root, text="AC (izpolni dB)", font=HZ_FONT).grid(row=2,column=0,sticky="W",columnspan=2,padx=4,pady=2)

ac_row = 3
lowest_row = ac_row
for i,f in enumerate(["125","250","500","1000","1500","2000","3000","4000","6000","8000"]):
    n_decimals = 3
    if i > 1:
        n_decimals = 1
    entries = root.add_entry(entries, f"{float(f)/1000:.{n_decimals}f} kHz", f"{f}BefAir",i+ac_row)
    lowest_row += 1

ttk.Label(root, text="BC (izpolni dB)", font=HZ_FONT).grid(row=2,column=2,sticky="W",columnspan=2, padx=4,pady=2)

for i,f in enumerate(["500","1000","1500","2000","3000","4000"]):
    entries = root.add_entry(entries,f"{float(f)/1000:.1f} kHz", f"{f}BefBone",i+ac_row,2)

pred_labels = {}


button_row = lowest_row
lowest_row +=1
ttk.Label(root, text="Predicted values", font=HZ_FONT).grid(row=lowest_row,column=0,sticky="W",columnspan=2)
lowest_row +=1
ttk.Label(root, text="AC", font=HZ_FONT).grid(row=lowest_row,column=0,sticky="W",columnspan=2,padx=4,pady=2)
ttk.Label(root, text="BC", font=HZ_FONT).grid(row=lowest_row,column=2,sticky="W",columnspan=2,padx=4,pady=2)
lowest_row +=1

for i,f in enumerate([0.125,0.25,0.5,1.0,1.5,2.0,3.0,4.0,6.0,8.0]):
    add_pred(root, f"{f} kHz", f"{f}pa",row=lowest_row+i)

for i,f in enumerate([0.5,1.0,1.5,2.0,3.0,4.0]):
    add_pred(root, f"{f} kHz", f"{f}pb",row=lowest_row+i,col=2)



fig_canvas = None

def on_ok():
    global fig_canvas

    observed_dict = {}
    print(entries)
    for k, e in entries.items():
        if len(e.get()) == 0 and k == "Spol (Z/M)":  #,"Stran glave (L/D
            observed_dict[k] = 0.65

        if len(e.get()) == 0 and k == "Stran glave (L/D)":  #,"Stran glave (L/D
            observed_dict[k] = 0.5
        
        try:
            observed_dict[k] = float(e.get())
        except:
            observed_dict[k] = 0.0
            
    if irregular_entries(observed_dict):
        return

    observed = pd.DataFrame(observed_dict, index=[0])
    observed_np = np.array(observed.iloc[0,2:])
    print(observed)
    print(observed_np)
    print(feature_means)
    print(feature_stds)
    #observed_norm = (observed - feature_means[1:]) / feature_stds[1:]

    predicted = np.zeros(16)

    for i in range(1, 17):
        with open(f"models/{i}", "rb") as f:
            model = pickle.load(f)

        tmp = model.predict(observed_norm[columns_list[i-1]])
        tmp = funky.roundto5s(tmp)
        predicted[i-1] = tmp[0]

        key = f"{FREQS[i-1]}pa" if i < 11 else f"{FREQS[i-1]}pb"
        pred_labels[key].config(text=str(int(tmp[0])))

    if fig_canvas:
        fig_canvas.get_tk_widget().destroy()

    fig = fig_maker(observed_np, predicted, X_GRID, Y_GRID, X_AIR, X_BONE)
    fig_canvas = draw_figure(root, fig)
    
# btns = ttk.Frame(root)
# btns.pack(pady=10)

ttk.Button(root, text="GENERIRAJ AUDIOGRAM", command=on_ok).grid(row=button_row,column=0,columnspan=2)
ttk.Button(root, text="Exit", command=root.destroy).grid(row=button_row,column=2,columnspan=2)

root.mainloop()
