# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import math
import re

import io
import os

from google.colab import drive
drive.mount('/content/drive', force_remount=True)

from google.colab import auth
auth.authenticate_user()
import requests
gcloud_token = !gcloud auth print-access-token
gcloud_tokeninfo = requests.get('https://www.googleapis.com/oauth2/v3/tokeninfo?access_token=' + gcloud_token[0]).json()
user = gcloud_tokeninfo['email'].split('@')[0]

if user == "amirf2":
    file_path = r"/content/drive/MyDrive/Documents/Physics/Lab 1A/Geometrical Optics/geometrical_optics_data.xlsx" # Replace with your file path: r"/content/<your file name>.xlsx"
if user == "itayfeldman1":
    file_path = r"/content/drive/MyDrive/Physics Lab/LabA1/GeometricalOptics/geometrical_optics_data.xlsx"

sheet_name = 1 # Replace with your sheet number
data = pd.read_excel(file_path, sheet_name=sheet_name)

f_inf, f_inf_error = data.iloc[:,14][0], data.iloc[:,15][0]
f_not_inf, f_not_inf_error = data.iloc[:,16][0], data.iloc[:,17][0]
H, H_error = data.iloc[:,12][0], data.iloc[:,13][0]
print(f_inf, f_inf_error)
print(f_not_inf, f_not_inf_error)
print(H, H_error)

data[:5]


def choose_columns(columns):
    x = data.iloc[:,columns[0]]
    delta_x = data.iloc[:,columns[1]]
    y = data.iloc[:,columns[2]]
    delta_y = data.iloc[:,columns[3]]
    return x, delta_x, y, delta_y



def print_output(fit_params, fit_params_error, chi2_red, p_val, degrees_of_freedom):
    for i in range(len(fit_params)):
        print(f"a[{i}]: {fit_params[i]} \u00B1 {fit_params_error[i]} ({(abs(fit_params_error[i]/fit_params[i]))*100}% error)")
    print(f"chi squared reduced = {chi2_red:.5f} \u00B1 {np.sqrt(2/degrees_of_freedom)}")
    print(f"p-probability = {p_val:.5e}")
    print(f"DOF = {degrees_of_freedom}")

"""<div dir="rtl" lang="he" xml:lang="he">

כעת, נבצע את כל תהליך ביצוע ההתאמה והחישובים הנלווים, בבת אחת, ע"י קריאה לכל הפונקציות שראינו לעיל. ראשית נבחר את העמודות הרלוונטיות ונשרטט את הדאטא בלבד (אנא עשו בדיקת שפיות: האם זו התצורה שציפיתם לקבל?):
"""



def fit(columns,
        fit_func,
        initial_guesses,
        parameter_dimensions,
        x_dimension,
        y_dimension,
        fit_title,
        residuals_title,
        calculate_value_functions=None,
        statistical_value_functions=None,
        extremes_to_remove=None):
    calculate_value_functions = list() if calculate_value_functions is None else calculate_value_functions
    statistical_value_functions = list() if statistical_value_functions is None else statistical_value_functions

    # Data
    x, delta_x, y, delta_y = choose_columns(columns)

    figure_num_start = 1
    if extremes_to_remove is not None:
        remove_extremes((x, delta_x, y, delta_y), extremes_to_remove)
        figure_num_start = 100

    # Data graph
    plt.close('all')
    fig, ax = plt.subplots(1, 1, figsize=(6, 6), num=figure_num_start)
    plt.style.use('classic')

    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    ax.errorbar(x, y, xerr=delta_x, yerr=delta_y, fmt='.b', label='Data', ecolor='gray') # Change the label if needed

    ax.set_title('Data - add here the full title')  # Add here the full title for the fit
    ax.set_xlabel(f'{data.columns[columns[0]]}') # Change x-axis label if needed
    ax.set_ylabel(f'{data.columns[columns[2]]}') # Change y-axis label if needed

    ax.grid(True)
    ax.legend()

    data_graph = io.BytesIO()
    fig.savefig(data_graph, bbox_inches='tight')
    #Fit
    fit_params, fit_params_error, fit_cov, output = odr_fit(fit_func, initial_guesses, x, delta_x, y, delta_y)
    residuals, degrees_of_freedom, chi2_red, p_val = calc_stats(x, y, fit_func, fit_params, output)
    print_output(fit_params, fit_params_error, chi2_red, p_val, degrees_of_freedom)

    # Fit Graph
    fig, axs = plt.subplots(1, 2, figsize=(15, 6), num=figure_num_start+1)
    plt.style.use('classic')

    fig.patch.set_facecolor('white')
    for ax in axs:
        ax.set_facecolor('white')

    x_fit = np.linspace(min(x), max(x), 10*len(x))
    y_fit = fit_func(fit_params, x_fit)

    axs[0].errorbar(x, residuals, xerr=delta_x, yerr=delta_y, fmt='.b', label="Data", ecolor='gray')
    axs[0].hlines(0, min(x), max(x), colors='r', linestyles='dashed')

    axs[0].set_title(residuals_title) # Add here the full title for the residuals
    axs[0].set_xlabel(f'{data.columns[columns[0]]} {x_dimension}') # Change column names if needed
    axs[0].set_ylabel(f'{data.columns[columns[2]]} - f({data.columns[columns[0]]}) {y_dimension}') # Change column names if needed

    axs[0].grid(True)
    axs[0].legend()

    axs[1].errorbar(x, y, xerr=delta_x, yerr=delta_y, fmt='.b', label='Data', ecolor='gray') # Change the label if needed
    axs[1].plot(x_fit, y_fit, label='Fit', c='r', alpha=0.5) # Change the label if needed

    axs[1].set_title(fit_title)  # Add here the full title for the fit
    axs[1].set_xlabel(f'{data.columns[columns[0]]} {x_dimension}') # Change x-axis label if needed
    axs[1].set_ylabel(f'{data.columns[columns[2]]} {y_dimension}') # Change y-axis label if needed

    axs[1].grid(True)
    axs[1].legend()

    fit_graph = io.BytesIO()
    fig.savefig(fit_graph, bbox_inches='tight')

    plt.tight_layout()
    plt.show()

    # Fit Parameters to Word.
    values = []
    for calculate_value_function in calculate_value_functions:
        values.append(calculate_value_function(fit_params, fit_params_error))

    for statistical_value_function in statistical_value_functions:
        values.append(statistical_value_function(fit_params, fit_params_error, values))

    doc = docx.Document('fit.docx') if os.path.exists('fit.docx') else docx.Document()

    row_count = len(fit_params) + 2 + len(values)
    tbl = doc.add_table(row_count, 2)
    tbl.alignment = docx.enum.table.WD_TABLE_ALIGNMENT.CENTER
    tbl.table_direction = docx.enum.table.WD_TABLE_DIRECTION.RTL
    tbl.autofit = True
    for i in range(len(fit_params)):
        add_math_to_tbl(tbl.rows[i], f'a_{i}', parameter_dimensions[i], fit_params[i], fit_params_error[i])
    add_math_to_tbl(tbl.rows[len(fit_params)], '\chi_{red}^2', None, chi2_red, np.sqrt(2/degrees_of_freedom))
    add_math_to_tbl(tbl.rows[len(fit_params)+1], 'P_{prob}', None, p_val)
    for i, value in enumerate(values):
        add_math_to_tbl(tbl.rows[len(fit_params)+2+i], value[0], value[1], value[2], value[3])

    doc.add_picture(data_graph)
    doc.add_picture(fit_graph)

    doc.save('fit.docx')
    data_graph.close()
    fit_graph.close()
    a = None
    if p_val > 0.95 or p_val < 0.05 and extremes_to_remove is None:
        # Try Removing extreme measurements
        print("Trying to remove extremes for better fit.")
        extreme_measurements = check_for_extreme_measurements(residuals, delta_y)
        print(extreme_measurements)
        a = fit(columns, fit_func, initial_guesses, parameter_dimensions, x_dimension, y_dimension,
            fit_title, residuals_title, calculate_value_functions, statistical_value_functions, extreme_measurements)
    if a is None:
        return x, delta_x, y, delta_y, fit_params, fit_params_error, values
    return a

"""<div dir="rtl" lang="he" xml:lang="he">

---
<h1>
<font size=5><b>התאמה 1 - נוסחת העדשות הדקות ליניארית</b></font>
</h1>

---
"""

columns = [0, 1, 2, 3] # Define the columns indices to represent x, delta x, y, delta y.
x, delta_x, y, delta_y = choose_columns(columns)

remove_extremes((x, delta_x, y, delta_y), [(7, 11.800203343131077), (2, 21.047728490477915), (3, 22.88370935912959)])

plt.close('all')
fig, ax = plt.subplots(1, 1, figsize=(6, 6))
plt.style.use('classic')

fig.patch.set_facecolor('white')
ax.set_facecolor('white')

ax.errorbar(x, y, xerr=delta_x, yerr=delta_y, fmt='.b', label='Data', ecolor='gray') # Change the label if needed

ax.set_title('Data - add here the full title')  # Add here the full title for the fit
ax.set_xlabel(f'{data.columns[columns[0]]}') # Change x-axis label if needed
ax.set_ylabel(f'{data.columns[columns[2]]}') # Change y-axis label if needed

ax.grid(True)
ax.legend()

plt.tight_layout()
plt.show()

fit_func = linear # Choose your fit function name
initial_guesses = (0, 0,) # Define the initial guesses for the parameters in list "A" (make sure they are the same length, and in the same order!)
fit_params, fit_params_error, fit_cov, output = odr_fit(fit_func, initial_guesses, x, delta_x, y, delta_y)
residuals, degrees_of_freedom, chi2_red, p_val = calc_stats(x, y, fit_func, fit_params, output)
print_output(fit_params, fit_params_error, chi2_red, p_val, degrees_of_freedom)

f, f_error = 1/fit_params[0], fit_params_error[0]/(fit_params[0]**2)
print(f"\nguessed f: {f} \u00B1 {f_error} ({(f_error/f)*100}% error)")

n_sigma = abs(f - f_inf) / math.sqrt((f_error)**2 + f_inf_error**2)
print(f"N sigma for f = {n_sigma}")

n_sigma = abs(fit_params[1] + 1) / math.sqrt(fit_params_error[1])
print(f"N sigma for a1 = {n_sigma}")

"""<div dir="rtl" lang="he" xml:lang="he">
<font size=5 color="red"> אופציונאלי:</font>
נדפיס את הפלט הסטנדרטי המלא, לקבלת מידע מלא יותר על תהליך ביצוע ההתאמה (לפירוט, ראו את הדוקומנטציה).

[דוקומנטציה](https://docs.scipy.org/doc/scipy/reference/generated/scipy.odr.Output.html)
"""

output.pprint()

"""<div dir="rtl" lang="he" xml:lang="he">

---
<h1>
<font size=5><b>7. שרטוט גרף הנתונים, פונקציית ההתאמה, וגרף השארים</b></font>
</h1>

נשרטט את ערכי הנתונים ושגיאותיהם (שימו לב שגם השגיאה בציר x משורטטת באיור זה, אם כי בשל מימדיה הקטנים היא לא נראית בבירור בגרף), ואת פונקציית ההתאמה שחישבנו ביחס אליהם, באיור השמאלי, ואילו את גרף השארים באיור הימני. תוכלו לשנות ולהתאים את פסקת הקוד הבאה לצרכיכם ולטעמכם, וכן להוסיף עוד עקומות במידת הנדרש. שימו לב שעליכם לערוך את הכותרות הראשיות לגרפים, כותרות הצירים, ולהוסיף יחידות.

---
"""

plt.close('all')
fig, axs = plt.subplots(1, 2, figsize=(15, 6))
plt.style.use('classic')

fig.patch.set_facecolor('white')
for ax in axs:
    ax.set_facecolor('white')

x_fit = np.linspace(min(x), max(x), 10*len(x))
y_fit = fit_func(fit_params, x_fit)
axs[0].errorbar(x, y, xerr=delta_x, yerr=delta_y, fmt='.b', label='Data', ecolor='gray') # Change the label if needed
axs[0].plot(x_fit, y_fit, label='Fit', c='r', alpha=0.5) # Change the label if needed

# If you want to plot multiple functions, change here the relevant parameters (x, y, xerr, yerr, label). Otherwise, uncomment the 2 next lines:
#axs[0].errorbar(x + 0.2, y + 0.3, xerr=delta_x, yerr=delta_y, fmt='.g', label='Data', ecolor='gray')
#axs[0].plot(x_fit + 0.2, y_fit + 0.3, label='Fit', c='k', alpha=0.5)

axs[0].set_title('Thin Lens Approximation - Linear Fit')  # Add here the full title for the fit
axs[0].set_xlabel(f'{data.columns[columns[0]]} [1/cm]]') # Change x-axis label if needed
axs[0].set_ylabel(f'{data.columns[columns[2]]} [1/cm]') # Change y-axis label if needed

axs[0].grid(True)
axs[0].legend()

axs[1].errorbar(x, residuals, xerr=delta_x, yerr=delta_y, fmt='.b', label="Data", ecolor='gray')
axs[1].hlines(0, min(x), max(x), colors='r', linestyles='dashed')

axs[1].set_title('Thin Lens Approximation - Linear Fit Residuals') # Add here the full title for the residuals
axs[1].set_xlabel(f'{data.columns[columns[0]]} [1/cm]') # Change column names if needed
axs[1].set_ylabel(f'{data.columns[columns[2]]} - f({data.columns[columns[0]]}) [1/cm]') # Change column names if needed

axs[1].grid(True)
axs[1].legend()

plt.tight_layout()
plt.show()

print(check_for_extreme_measurements(residuals, delta_y))

columns = [4, 5, 6, 7] # Define the columns indices to represent x, delta x, y, delta y.
x, delta_x, y, delta_y = choose_columns(columns)

remove_extremes((x, delta_x, y, delta_y), [(6, 11.685203558189603), (3, 19.978762542781155), (2, 21.322929385474538)])

plt.close('all')
fig, ax = plt.subplots(1, 1, figsize=(6, 6))
plt.style.use('classic')

fig.patch.set_facecolor('white')
ax.set_facecolor('white')

ax.errorbar(x, y, xerr=delta_x, yerr=delta_y, fmt='.b', label='Data', ecolor='gray') # Change the label if needed


ax.set_title('Data - add here the full title')  # Add here the full title for the fit
ax.set_xlabel(f'{data.columns[columns[0]]}') # Change x-axis label if needed
ax.set_ylabel(f'{data.columns[columns[2]]}') # Change y-axis label if needed

ax.grid(True)
ax.legend()

plt.tight_layout()
plt.show()

fit_func = optics # Choose your fit function name
initial_guesses = (0, f_inf,) # Define the initial guesses for the parameters in list "A" (make sure they are the same length, and in the same order!)
fit_params, fit_params_error, fit_cov, output = odr_fit(fit_func, initial_guesses, x, delta_x, y, delta_y)
residuals, degrees_of_freedom, chi2_red, p_val = calc_stats(x, y, fit_func, fit_params, output)
print_output(fit_params, fit_params_error, chi2_red, p_val, degrees_of_freedom)

print(f"\nguessed f: {fit_params[1]} \u00B1 {fit_params_error[1]} ({(fit_params_error[1]/fit_params[1])*100}% error)")

n_sigma = abs(fit_params[1] - f_inf) / math.sqrt((fit_params_error[1])**2 + f_inf_error**2)
print(f"N sigma f to f_inf = {n_sigma}")

n_sigma = abs(f - fit_params[1]) / math.sqrt((f_error)**2 + fit_params_error[1]**2)
print(f"N sigma f to linear f = {n_sigma}")

print(f"a0: {fit_params[0]}, v_min {min(y)}, propotion: {min(y)/abs(fit_params[0])}")

"""<div dir="rtl" lang="he" xml:lang="he">
<font size=5 color="red"> אופציונאלי:</font>
נדפיס את הפלט הסטנדרטי המלא, לקבלת מידע מלא יותר על תהליך ביצוע ההתאמה (לפירוט, ראו את הדוקומנטציה).

[דוקומנטציה](https://docs.scipy.org/doc/scipy/reference/generated/scipy.odr.Output.html)
"""

output.pprint()

"""<div dir="rtl" lang="he" xml:lang="he">

---
<h1>
<font size=5><b>7. שרטוט גרף הנתונים, פונקציית ההתאמה, וגרף השארים</b></font>
</h1>

נשרטט את ערכי הנתונים ושגיאותיהם (שימו לב שגם השגיאה בציר x משורטטת באיור זה, אם כי בשל מימדיה הקטנים היא לא נראית בבירור בגרף), ואת פונקציית ההתאמה שחישבנו ביחס אליהם, באיור השמאלי, ואילו את גרף השארים באיור הימני. תוכלו לשנות ולהתאים את פסקת הקוד הבאה לצרכיכם ולטעמכם, וכן להוסיף עוד עקומות במידת הנדרש. שימו לב שעליכם לערוך את הכותרות הראשיות לגרפים, כותרות הצירים, ולהוסיף יחידות.

---
"""

plt.close('all')
fig, axs = plt.subplots(1, 2, figsize=(15, 6))
plt.style.use('classic')

fig.patch.set_facecolor('white')
for ax in axs:
    ax.set_facecolor('white')

x_fit = np.linspace(min(x), max(x), 10*len(x))
y_fit = fit_func(fit_params, x_fit)
axs[0].errorbar(x, y, xerr=delta_x, yerr=delta_y, fmt='.b', label='Data', ecolor='gray') # Change the label if needed
axs[0].plot(x_fit, y_fit, label='Fit', c='r', alpha=0.5) # Change the label if needed

# If you want to plot multiple functions, change here the relevant parameters (x, y, xerr, yerr, label). Otherwise, uncomment the 2 next lines:
#axs[0].errorbar(x + 0.2, y + 0.3, xerr=delta_x, yerr=delta_y, fmt='.g', label='Data', ecolor='gray')
#axs[0].plot(x_fit + 0.2, y_fit + 0.3, label='Fit', c='k', alpha=0.5)

axs[0].set_title('Thin Lens Approximation - Non-Linear Fit')  # Add here the full title for the fit
axs[0].set_xlabel(f'{data.columns[columns[0]]} [cm]') # Change x-axis label if needed
axs[0].set_ylabel(f'{data.columns[columns[2]]} [cm]') # Change y-axis label if needed

axs[0].grid(True)
axs[0].legend()

axs[1].errorbar(x, residuals, xerr=delta_x, yerr=delta_y, fmt='.b', label="Data", ecolor='gray')
axs[1].hlines(0, min(x), max(x), colors='r', linestyles='dashed')

axs[1].set_title('Thin Lens Approximation - Non-Linear Fit Residuals') # Add here the full title for the residuals
axs[1].set_xlabel(f'{data.columns[columns[0]]} [cm]') # Change column names if needed
axs[1].set_ylabel(f'{data.columns[columns[2]]} - f({data.columns[columns[0]]}) [cm]') # Change column names if needed

axs[1].grid(True)
axs[1].legend()

plt.tight_layout()
plt.show()

print(check_for_extreme_measurements(residuals, delta_y))

"""<div dir="rtl" lang="he" xml:lang="he">

---
<h1>
<font size=5><b>התאמה 3 - נוסחת ההגדלה ליניארית</b></font>
</h1>

---
"""

columns = [8, 9, 10, 11] # Define the columns indices to represent x, delta x, y, delta y.
x, delta_x, y, delta_y = choose_columns(columns)

plt.close('all')
fig, ax = plt.subplots(1, 1, figsize=(6, 6))
plt.style.use('classic')

fig.patch.set_facecolor('white')
ax.set_facecolor('white')

ax.errorbar(x, y, xerr=delta_x, yerr=delta_y, fmt='.b', label='Data', ecolor='gray') # Change the label if needed


ax.set_title('Data - add here the full title')  # Add here the full title for the fit
ax.set_xlabel(f'{data.columns[columns[0]]}') # Change x-axis label if needed
ax.set_ylabel(f'{data.columns[columns[2]]}') # Change y-axis label if needed

ax.grid(True)
ax.legend()

plt.tight_layout()
plt.show()

fit_func = linear # Choose your fit function name
initial_guesses = (0, 0,) # Define the initial guesses for the parameters in list "A" (make sure they are the same length, and in the same order!)
fit_params, fit_params_error, fit_cov, output = odr_fit(fit_func, initial_guesses, x, delta_x, y, delta_y)
residuals, degrees_of_freedom, chi2_red, p_val = calc_stats(x, y, fit_func, fit_params, output)
print_output(fit_params, fit_params_error, chi2_red, p_val, degrees_of_freedom)

n_sigma = abs(H - fit_params[1]) / math.sqrt((H_error)**2 + fit_params_error[1]**2)
print(f"n-sigma H to -a_1 = {n_sigma}")

"""<div dir="rtl" lang="he" xml:lang="he">
<font size=5 color="red"> אופציונאלי:</font>
נדפיס את הפלט הסטנדרטי המלא, לקבלת מידע מלא יותר על תהליך ביצוע ההתאמה (לפירוט, ראו את הדוקומנטציה).

[דוקומנטציה](https://docs.scipy.org/doc/scipy/reference/generated/scipy.odr.Output.html)
"""

output.pprint()

"""<div dir="rtl" lang="he" xml:lang="he">

---
<h1>
<font size=5><b>7. שרטוט גרף הנתונים, פונקציית ההתאמה, וגרף השארים</b></font>
</h1>

נשרטט את ערכי הנתונים ושגיאותיהם (שימו לב שגם השגיאה בציר x משורטטת באיור זה, אם כי בשל מימדיה הקטנים היא לא נראית בבירור בגרף), ואת פונקציית ההתאמה שחישבנו ביחס אליהם, באיור השמאלי, ואילו את גרף השארים באיור הימני. תוכלו לשנות ולהתאים את פסקת הקוד הבאה לצרכיכם ולטעמכם, וכן להוסיף עוד עקומות במידת הנדרש. שימו לב שעליכם לערוך את הכותרות הראשיות לגרפים, כותרות הצירים, ולהוסיף יחידות.

---
"""

plt.close('all')
fig, axs = plt.subplots(1, 2, figsize=(15, 6))
plt.style.use('classic')

fig.patch.set_facecolor('white')
for ax in axs:
    ax.set_facecolor('white')

x_fit = np.linspace(min(x), max(x), 10*len(x))
y_fit = fit_func(fit_params, x_fit)
axs[0].errorbar(x, y, xerr=delta_x, yerr=delta_y, fmt='.b', label='Data', ecolor='gray') # Change the label if needed
axs[0].plot(x_fit, y_fit, label='Fit', c='r', alpha=0.5) # Change the label if needed

# If you want to plot multiple functions, change here the relevant parameters (x, y, xerr, yerr, label). Otherwise, uncomment the 2 next lines:
#axs[0].errorbar(x + 0.2, y + 0.3, xerr=delta_x, yerr=delta_y, fmt='.g', label='Data', ecolor='gray')
#axs[0].plot(x_fit + 0.2, y_fit + 0.3, label='Fit', c='k', alpha=0.5)

axs[0].set_title('Magnification Factor Equation - Linear Fit')  # Add here the full title for the fit
axs[0].set_xlabel(f'{data.columns[columns[0]]} [cm/cm]') # Change x-axis label if needed
axs[0].set_ylabel(f'{data.columns[columns[2]]} [cm]') # Change y-axis label if needed

axs[0].grid(True)
axs[0].legend()

axs[1].errorbar(x, residuals, xerr=delta_x, yerr=delta_y, fmt='.b', label="Data", ecolor='gray')
axs[1].hlines(0, min(x), max(x), colors='r', linestyles='dashed')

axs[1].set_title('Magnification Factor Equation - Linear Fit Residuals') # Add here the full title for the residuals
axs[1].set_xlabel(f'{data.columns[columns[0]]} [cm/cm]') # Change column names if needed
axs[1].set_ylabel(f'{data.columns[columns[2]]} - fit({data.columns[columns[0]]}) [cm]') # Change column names if needed

axs[1].grid(True)
axs[1].legend()

plt.tight_layout()
plt.show()

def calculate_linear_f(fit_params, fit_params_error):
  return "f", "cm", 1/fit_params[0], fit_params_error[0]/(fit_params[0]**2)

nsigmas = [nsigma_with_outer_value("finf", f_inf, f_inf_error, None, 0),
           nsigma_with_outer_value("a1", -1, 0, 1, None)]
x, delta_x, y, delta_y, fit_params, fit_params_error, values = fit([0,1,2,3], linear, (0,0), ("1/cm", None), "[1/cm]", "[1/cm]",
    'Thin Lens Approximation - Linear Fit', "Thin Lens Approximation - Linear Fit Residuals",
    [calculate_linear_f], nsigmas)
f_linear, f_linear_error = values[0][2], values[0][3]

def calculate_non_linear_f(fit_params, fit_params_error):
    return "f", "cm", fit_params[1], fit_params_error[1]

nsigmas = [nsigma_with_outer_value("finf", f_inf, f_inf_error, None, 0),
           nsigma_with_outer_value("prev", f_linear, f_linear_error, None, 0)]

x, delta_x, y, delta_y, fit_params, fit_params_error, values = fit([4,5,6,7], optics, (0, f_inf,), ("cm", "cm"), "[cm]", "[cm]",
                                                                   "Thin Lens Approximation - Non-Linear Fit", "Thin Lens Approximation - Non-Linear Fit Residuals",
                                                                   [calculate_non_linear_f], nsigmas)

print(f"a0: {fit_params[0]}, v_min {min(y)}, propotion: {min(y)/abs(fit_params[0])}")

def calcuate_H(fit_params, fit_params_error):
  return "H", "cm", fit_params[1], fit_params_error[1]

nsigmas = [nsigma_with_outer_value("H", H, H_error, None, 0)]
x, delta_x, y, delta_y, fit_params, fit_params_error, values = fit([8,9,10,11], linear, (0, 0,), ("cm", "cm"), "[cm/cm]", "[cm]",
    "Magnification Factor Equation - Linear Fit", "Magnification Factor Equation - Linear Fit Residuals",
    [calcuate_H], nsigmas)

print(f"a0: {fit_params[0]}, v_min {min(y)}, propotion: {min(y)/abs(fit_params[0])}")