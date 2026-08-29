import tkinter as tk
from tkinter import ttk
from tkinter import messagebox 
import pickle
from funct import load_pickle, add_constructed_features
import numpy as np
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from constants import FQ_FEATURES, FEATURES_NUM

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
# file = open('variables/feature_means','rb')
# feature_means = np.load(file)
# file.close()
# file = open('variables/feature_stds','rb')
# feature_stds = np.load(file)
# file.close()

models = load_pickle('models/best_models_cv_fitted')

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

adg_data = pd.read_csv('variables/processed.csv')


X = adg_data.iloc[:,:FEATURES_NUM]
cols_order = X.columns

def fig_maker(observed, predicted, x_grid, y_grid, x_air, x_bone):
    plt.close('all')

    fig, ax = plt.subplots(figsize=(8, 8))
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

def interpolate_values(frequency_cols: list[str], observed: dict, cnt_filled_limit: int):
    missing_sequences = []
    missing_sequence = []
    cnt_filled = 0
    for fq in frequency_cols:
        if observed[fq] != 0:
            if len(missing_sequence) > 0:
                missing_sequences.append(missing_sequence)
            missing_sequence = []
            cnt_filled += 1
            continue
        missing_sequence.append(fq)
    if len(missing_sequence) > 0:
        missing_sequences.append(missing_sequence)

    if cnt_filled < cnt_filled_limit:
        return cnt_filled_limit - cnt_filled
    
    for missing_sequence in missing_sequences:
        if len(missing_sequence) == 1:
            if frequency_cols.index(missing_sequence[0]) == 0:
                observed[missing_sequence[0]] = observed[frequency_cols[1]]
                continue
            if frequency_cols.index(missing_sequence[0]) == len(frequency_cols) -1:
                observed[missing_sequence[0]] = observed[frequency_cols[len(frequency_cols)-2]]
                continue
        if frequency_cols.index(missing_sequence[0]) == 0:
            index = frequency_cols.index(missing_sequence[len(missing_sequence)-1])+1
            freq = frequency_cols[index]
            filler_value = observed[freq]
        elif frequency_cols.index(missing_sequence[len(missing_sequence) - 1]) == len(frequency_cols) - 1:
            index = frequency_cols.index(missing_sequence[0])-1
            freq = frequency_cols[index]
            filler_value = observed[freq]
        else:
            r_index = frequency_cols.index(missing_sequence[len(missing_sequence)-1])+1
            l_index = frequency_cols.index(missing_sequence[0])-1
            filler_value = (observed[frequency_cols[l_index]] + observed[frequency_cols[r_index]]) / 2
        for el in missing_sequence:
            observed[el] = int(funky.roundto5s(filler_value))

    return observed

def error_exit(num_missing:int, suffix:str) -> str:
    if num_missing == 1:
        polje = "polje"
    elif num_missing == 2:
        polje = "polji"
    else:
        polje = "polj"
    return f"Izpolni še vsaj {num_missing} {polje} pri {suffix} vrednostih."

def interpolate_fill(observed:dict):
    air_fq_features = FQ_FEATURES[:10]
    bone_fq_features = FQ_FEATURES[10:]
    observed = interpolate_values(air_fq_features, observed, 3)
    if isinstance(observed, int):
        return error_exit(observed, "AC")
           
    observed = interpolate_values(bone_fq_features, observed, 2)
    if isinstance(observed, int):
        return error_exit(observed, "BC")

    return observed


def draw_figure(container, figure):
    canvas = FigureCanvasTkAgg(figure, master=container)
    canvas.draw()
    canvas.get_tk_widget().grid(row=0,column=10,rowspan=30,columnspan=2,padx=2,pady=2)
    return canvas

def add_entry(parent, label, key,row,col=0):
    ttk.Label(parent, text=label, font=HZ_FONT).grid(row=row,column=col,pady=4)
    e = ttk.Entry(parent, width=6, font=HZ_FONT)
    e.grid(row = row, column = col+1, pady = 2)
    entries[key] = e

def add_pred(parent, label, key,row,col=0):
    ttk.Label(parent, text=label, font=HZ_FONT).grid(row=row,column=col,pady=4)
    l = ttk.Label(parent, text="__", font=HZ_FONT, foreground="orange")
    l.grid(row = row, column = col+1, pady = 2)
    pred_labels[key] = l

root = tk.Tk()
root.title("Audiogram prediction (after otosclerosis surgery)")
root.geometry("1200x900")

entries = {}



ttk.Label(root, text="Patient age", font=HZ_FONT).grid(row=0,column=0,pady=2)
e = ttk.Entry(root, width=6, font=HZ_FONT)
e.grid(row=0,column=1,pady=2)
entries["Age"] = e
ttk.Label(root, text="Gender (0=M, 1=F)", font=HZ_FONT).grid(row=0,column=2,pady=2)
e = ttk.Entry(root, width=6, font=HZ_FONT)
e.grid(row=0,column=3,pady=2)
entries["Gender-0=M, 1 =F"] = e
ttk.Label(root, text="Side (0=L, 1=R)", font=HZ_FONT).grid(row=1,column=0,pady=2)
e = ttk.Entry(root, width=6, font=HZ_FONT)
e.grid(row=1,column=1,pady=2)
entries["Side 0=L, 1=R"] = e


ttk.Label(root, text="Air conduction values", font=HZ_FONT).grid(row=2,column=0,sticky="W",columnspan=2,padx=4,pady=2)

ac_row = 3
lowest_row = ac_row
for i,f in enumerate(["125","250","500","1000","1500","2000","3000","4000","6000","8000"]):
    label = f"{float(f)/1000:.3f} kHz"
    if f in ["500", "1000", "2000"]:
        label += "*"
    add_entry(root, label, f"{f}BefAir",i+ac_row)
    lowest_row += 1

ttk.Label(root, text="Bone conduction values", font=HZ_FONT).grid(row=2,column=2,sticky="W",columnspan=2, padx=4,pady=2)

for i,f in enumerate(["500","1000","1500","2000","3000","4000"]):
    label = f"{float(f)/1000:.3f} kHz"
    if f in ["500", "1000", "2000"]:
        label += "*"
    add_entry(root, label, f"{f}BefBone",i+ac_row,2)

pred_labels = {}


button_row = lowest_row
lowest_row +=1
ttk.Label(root, text="Predicted values", font=HZ_FONT).grid(row=lowest_row,column=0,sticky="W",columnspan=2)
lowest_row +=1

for i,f in enumerate([0.125,0.25,0.5,1.0,1.5,2.0,3.0,4.0,6.0,8.0]):
    add_pred(root, f"{f} kHz", f"{f}pa",row=lowest_row+i)

for i,f in enumerate([0.5,1.0,1.5,2.0,3.0,4.0]):
    add_pred(root, f"{f} kHz", f"{f}pb",row=lowest_row+i,col=2)



fig_canvas = None

def on_ok():
    global fig_canvas

    observed_dict = {}
    for k, e in entries.items():
        print(k,e, e.get())
        try:
            observed_dict[k] = float(e.get())
        except:
            observed_dict[k] = 0.0

        if len(e.get()) == 0 and k == "Age":  
            observed_dict[k] = 50
        if len(e.get()) == 0 and k == "Gender-0=M, 1 =F":  
            observed_dict[k] = 0.64
        if len(e.get()) == 0 and k == "Side 0=L, 1=R":  
            observed_dict[k] = 0.46

    observed_dict = interpolate_fill(observed_dict)
    if isinstance(observed_dict, str):
        messagebox.showwarning("showwarning", observed_dict) 
        return


    observed = pd.DataFrame(observed_dict, index=[0])
    observed = add_constructed_features(observed)


    #observed_np = np.array(observed.iloc[0,1:])
    print(observed)
    #print(observed_np)
 
    # observed_norm = (observed - feature_means[1:18]) / feature_stds[1:18]
    # print(observed_norm)
    # observed_norm = observed

    observed = observed[cols_order]

    print(observed)
    breakpoint()

    predicted = np.zeros(16)

    for i in range(16):
        model = models[i]

        tmp = model.predict(observed_norm[columns_list[i-1]])
        tmp = funky.roundto5s(tmp)
        predicted[i-1] = tmp[0]

        key = f"{FREQS[i-1]}pa" if i < 11 else f"{FREQS[i-1]}pb"
        pred_labels[key].config(text=str(int(tmp[0])))

    if fig_canvas:
        fig_canvas.get_tk_widget().destroy()
    print(observed_np)
    print(predicted)

    fig = fig_maker(observed_np, predicted, X_GRID, Y_GRID, X_AIR, X_BONE)
    fig_canvas = draw_figure(root, fig)
    
# btns = ttk.Frame(root)
# btns.pack(pady=10)

ttk.Button(root, text="GENERIRAJ AUDIOGRAM", command=on_ok).grid(row=button_row,column=0,columnspan=2)
ttk.Button(root, text="Exit", command=root.destroy).grid(row=button_row,column=2,columnspan=2)

root.mainloop()