"""Gas hydrate phase-equilibrium hybrid model (vdW-P physics + ML residual correction)."""

import pandas as pd
import numpy as np
import os
import sys
import json
from math import sqrt, exp
from datetime import datetime
from collections import Counter
import warnings
import platform
from scipy.optimize import minimize_scalar
from scipy.integrate import quad
from sklearn.model_selection import GridSearchCV, train_test_split, GroupKFold
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, max_error
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
import lightgbm as lgb
import joblib
from prettytable import PrettyTable
from tqdm import tqdm
_all_bips_df = None

def create_parameter_template(filename='hydrate_parameters.xlsx'):
    components_data = {'Name': ['Methane', 'Ethane', 'Propane', 'Carbon dioxide', 'Nitrogen', 'Hydrogen Sulfide', 'i-Butane'], 'Formula': ['CH4', 'C2H6', 'C3H8', 'CO2', 'N2', 'H2S', 'i-C4H10'], 'Tc': [190.56, 305.32, 369.83, 304.13, 126.2, 373.5, 408.1], 'Pc': [4.599, 4.872, 4.248, 7.377, 3.394, 9.008, 3.648], 'Acentric': [0.011, 0.099, 0.152, 0.225, 0.04, 0.081, 0.177], 'Structure': [1, 1, 2, 1, 2, 2, 2], 'Vc': [98.6, 145.5, 200.0, 94.0, 89.2, 98.5, 262.7]}
    components_df = pd.DataFrame(components_data)
    kihara_data = {'Name': ['Methane', 'Ethane', 'Propane', 'Carbon dioxide', 'Nitrogen', 'Hydrogen Sulfide', 'i-Butane'], 'Formula': ['CH4', 'C2H6', 'C3H8', 'CO2', 'N2', 'H2S', 'i-C4H10'], 'Core_Radius': [0.3834, 0.5651, 0.6502, 0.6805, 0.3526, 0.36, 0.8706], 'Sigma': [3.165, 3.24693, 3.3093, 2.9818, 3.0124, 3.153, 3.0822], 'Epsilon': [154.54, 174.97, 203.31, 168.77, 125.15, 204.85, 225.16]}
    kihara_df = pd.DataFrame(kihara_data)
    bip_data = [{'Comp1': 'CO2', 'Comp2': 'Methane', 'A_ij': 0.1, 'B_ij': 0.0}, {'Comp1': 'CO2', 'Comp2': 'Ethane', 'A_ij': 0.1298, 'B_ij': 0.0}, {'Comp1': 'CO2', 'Comp2': 'Nitrogen', 'A_ij': -0.02, 'B_ij': 0.0}, {'Comp1': 'CO2', 'Comp2': 'Hydrogen Sulfide', 'A_ij': 0.1, 'B_ij': 0.0}, {'Comp1': 'CO2', 'Comp2': 'i-Butane', 'A_ij': 0.1298, 'B_ij': 0.0}, {'Comp1': 'CO2', 'Comp2': 'Propane', 'A_ij': 0.135, 'B_ij': 0.0}, {'Comp1': 'Methane', 'Comp2': 'Ethane', 'A_ij': 0.00224, 'B_ij': 0.0}, {'Comp1': 'Methane', 'Comp2': 'Nitrogen', 'A_ij': 0.036, 'B_ij': 0.0}, {'Comp1': 'Methane', 'Comp2': 'Hydrogen Sulfide', 'A_ij': 0.085, 'B_ij': 0.0}, {'Comp1': 'Methane', 'Comp2': 'i-Butane', 'A_ij': 0.01311, 'B_ij': 0.0}, {'Comp1': 'Methane', 'Comp2': 'Propane', 'A_ij': 0.00683, 'B_ij': 0.0}, {'Comp1': 'Ethane', 'Comp2': 'Nitrogen', 'A_ij': 0.05, 'B_ij': 0.0}, {'Comp1': 'Ethane', 'Comp2': 'Hydrogen Sulfide', 'A_ij': 0.084, 'B_ij': 0.0}, {'Comp1': 'Ethane', 'Comp2': 'i-Butane', 'A_ij': 0.00457, 'B_ij': 0.0}, {'Comp1': 'Ethane', 'Comp2': 'Propane', 'A_ij': 0.00126, 'B_ij': 0.0}, {'Comp1': 'Nitrogen', 'Comp2': 'Hydrogen Sulfide', 'A_ij': 0.1676, 'B_ij': 0.0}, {'Comp1': 'Nitrogen', 'Comp2': 'i-Butane', 'A_ij': 0.095, 'B_ij': 0.0}, {'Comp1': 'Nitrogen', 'Comp2': 'Propane', 'A_ij': 0.08, 'B_ij': 0.0}, {'Comp1': 'Hydrogen Sulfide', 'Comp2': 'i-Butane', 'A_ij': 0.05, 'B_ij': 0.0}, {'Comp1': 'Hydrogen Sulfide', 'Comp2': 'Propane', 'A_ij': 0.075, 'B_ij': 0.0}, {'Comp1': 'i-Butane', 'Comp2': 'Propane', 'A_ij': 0.00104, 'B_ij': 0.0}]
    bip_df = pd.DataFrame(bip_data)
    bip_df_sym = pd.DataFrame([{'Comp1': row['Comp2'], 'Comp2': row['Comp1'], 'A_ij': row['A_ij'], 'B_ij': row['B_ij']} for _, row in bip_df.iterrows()])
    bip_df = pd.concat([bip_df, bip_df_sym]).drop_duplicates().reset_index(drop=True)
    try:
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            components_df.to_excel(writer, sheet_name='Components', index=False)
            kihara_df.to_excel(writer, sheet_name='Kihara', index=False)
            bip_df.to_excel(writer, sheet_name='BIPs', index=False)
        print(f'Created parameter template: {filename}')
        return True
    except Exception as e:
        print(f'Failed to create parameter template: {e}')
        return False

def load_component_data(component_name, params_file='hydrate_parameters.xlsx'):
    actual_name = component_name
    try:
        params = pd.read_excel(params_file, sheet_name='Components')
        kihara_params = pd.read_excel(params_file, sheet_name='Kihara')
        component_data = params[params['Name'].str.lower() == component_name.lower()]
        if component_data.empty:
            component_data = params[params['Name'].str.lower().str.contains(component_name.lower())]
            if component_data.empty:
                raise ValueError(f"Component '{component_name}' not found.")
            if len(component_data) > 1:
                component_data = component_data.iloc[[0]]
        actual_name = component_data['Name'].values[0]
        tc, pc, acentric, structure = (component_data['Tc'].values[0], component_data['Pc'].values[0], component_data['Acentric'].values[0], component_data['Structure'].values[0])
        kihara_data = kihara_params[kihara_params['Name'].str.lower() == actual_name.lower()]
        if kihara_data.empty:
            raise ValueError(f"Kihara parameters for '{actual_name}' not found in the Excel file.")
        else:
            kihara_parameters = np.array([[kihara_data['Core_Radius'].values[0]], [kihara_data['Sigma'].values[0]], [kihara_data['Epsilon'].values[0]]])
        return {'name': actual_name, 'tc': tc, 'pc': pc, 'acentric': acentric, 'structure': structure, 'kihara_params': kihara_parameters}
    except Exception as e:
        raise ValueError(f"Failed to load data for '{component_name}'. Check template file. Error: {e}")

def get_bip(comp1_name, comp2_name, temperature):
    global _all_bips_df
    if comp1_name == comp2_name:
        return 0.0
    if _all_bips_df is None or _all_bips_df.empty:
        return 0.0
    if temperature <= 1e-06:
        return 0.0
    bip_row = _all_bips_df[(_all_bips_df['Comp1'].str.lower() == comp1_name.lower()) & (_all_bips_df['Comp2'].str.lower() == comp2_name.lower())]
    if not bip_row.empty:
        return bip_row['A_ij'].values[0] + bip_row['B_ij'].values[0] / temperature
    bip_row = _all_bips_df[(_all_bips_df['Comp1'].str.lower() == comp2_name.lower()) & (_all_bips_df['Comp2'].str.lower() == comp1_name.lower())]
    if not bip_row.empty:
        return bip_row['A_ij'].values[0] + bip_row['B_ij'].values[0] / temperature
    return 0.0

def pr_eos(P, T, components_data, z):
    R = 8.3144598
    num_components = len(components_data)
    omega_a_pr = 0.45724
    omega_b_pr = 0.0778
    a_pure_i, b_pure_i, alpha_i, m_i = (np.zeros(num_components) for _ in range(4))
    for i in range(num_components):
        Tc_i, Pc_i, omega_i = (components_data[i]['tc'], components_data[i]['pc'], components_data[i]['acentric'])
        Tr_i = T / Tc_i
        if Tr_i <= 0:
            raise ValueError('Reduced temperature <= 0.')
        a_pure_i[i] = omega_a_pr * (R * Tc_i) ** 2 / Pc_i
        b_pure_i[i] = omega_b_pr * R * Tc_i / Pc_i
        m_i[i] = 0.379642 + 1.48503 * omega_i - 0.164423 * omega_i ** 2 + 0.016666 * omega_i ** 3
        alpha_i[i] = (1 + m_i[i] * (1 - np.sqrt(Tr_i))) ** 2
    k_ij = np.zeros((num_components, num_components))
    for i in range(num_components):
        for j in range(num_components):
            k_ij[i, j] = get_bip(components_data[i]['name'], components_data[j]['name'], T)
    a_mix = sum((z[i] * z[j] * np.sqrt(a_pure_i[i] * alpha_i[i] * a_pure_i[j] * alpha_i[j]) * (1 - k_ij[i, j]) for i in range(num_components) for j in range(num_components)))
    b_mix = np.sum(z * b_pure_i)
    A_mix = a_mix * P / (R * T) ** 2
    B_mix = b_mix * P / (R * T)
    coeffs = [1, B_mix - 1, A_mix - 3 * B_mix ** 2 - 2 * B_mix, B_mix ** 2 + B_mix ** 3 - A_mix * B_mix]
    roots = np.roots(coeffs)
    real_positive_roots = [r.real for r in roots if np.isreal(r) and r.real > 0]
    if not real_positive_roots:
        raise ValueError('No valid real positive Z found.')
    Z_vap = max(real_positive_roots)
    phi_i = np.zeros(num_components)
    log_num_arg = Z_vap + (1 + np.sqrt(2)) * B_mix
    log_den_arg = Z_vap + (1 - np.sqrt(2)) * B_mix
    if log_num_arg <= 1e-10 or log_den_arg <= 1e-10:
        raise ValueError('Log argument non-positive.')
    log_ratio = np.log(log_num_arg / log_den_arg)
    common_term3_coeff = A_mix / (2 * np.sqrt(2) * B_mix) if B_mix > 1e-10 else 0.0
    for i in range(num_components):
        term1_ln_phi = b_pure_i[i] / b_mix * (Z_vap - 1) if b_mix > 1e-10 else 0.0
        if Z_vap - B_mix <= 1e-10:
            raise ValueError('Log(Z-B) argument non-positive.')
        term2_ln_phi = -np.log(Z_vap - B_mix)
        sum_term_for_partial_A = sum((z[j] * np.sqrt(a_pure_i[i] * alpha_i[i] * a_pure_i[j] * alpha_i[j]) * (1 - k_ij[i, j]) for j in range(num_components)))
        bracket_term3 = 2 * sum_term_for_partial_A / a_mix - b_pure_i[i] / b_mix if a_mix > 1e-10 and b_mix > 1e-10 else 0.0
        term3_ln_phi = -common_term3_coeff * bracket_term3 * log_ratio
        ln_phi_i_val = term1_ln_phi + term2_ln_phi + term3_ln_phi
        phi_i[i] = np.exp(np.clip(ln_phi_i_val, -50, 50))
    return (Z_vap, phi_i)

def calculate_mixture_fugacities(pressure, temperature, components_data, composition):
    if pressure <= 0:
        pressure = 0.0001
    Z_vap, phi_i = pr_eos(pressure, temperature, components_data, np.array(composition))
    return phi_i * np.array(composition) * pressure

def kihara_mix(temperature, structure, potential_parameters):
    k_B = 1.38064852e-23
    if structure == 1:
        cavity_radius = np.array([3.95e-10, 4.33e-10])
        z_ljp = np.array([20, 24])
    elif structure == 2:
        cavity_radius = np.array([3.91e-10, 4.73e-10])
        z_ljp = np.array([20, 28])
    else:
        raise ValueError(f'Invalid structure type: {structure}.')
    c_mpa_out = np.zeros((potential_parameters.shape[1], 2))
    for i in range(potential_parameters.shape[1]):
        core_radius = potential_parameters[0, i] * 1e-10
        sigma = potential_parameters[1, i] * 1e-10
        epsilon = potential_parameters[2, i]
        for j in range(2):

            def del_fn(r, n):
                if r < 0 or r > cavity_radius[j] - core_radius:
                    return 0.0
                term_a = np.clip(1 - r / cavity_radius[j] - core_radius / cavity_radius[j], 1e-15, None)
                term_b = np.clip(1 + r / cavity_radius[j] - core_radius / cavity_radius[j], 1e-15, None)
                return 1 / n * (term_a ** (-n) - term_b ** (-n))

            def w_fn(r):
                r = max(r, 1e-15)
                term1 = sigma ** 12 / (cavity_radius[j] ** 11 * r) * (del_fn(r, 10) + core_radius * del_fn(r, 11) / cavity_radius[j])
                term2 = sigma ** 6 / (cavity_radius[j] ** 5 * r) * (del_fn(r, 4) + core_radius * del_fn(r, 5) / cavity_radius[j])
                return np.clip(2 * z_ljp[j] * epsilon * (term1 - term2), -700 * temperature, 700 * temperature)

            def c_trm(r):
                return np.exp(-w_fn(r) / temperature) * r ** 2
            upper_limit = cavity_radius[j] - core_radius
            if upper_limit <= 1e-15:
                c_mpa_out[i, j] = 0.0
                continue
            try:
                integration_result, _ = quad(c_trm, 0, upper_limit, limit=50)
                c_mpa_out[i, j] = np.clip(4 * np.pi * integration_result * 1000000.0 / (k_B * temperature), 0.0, 10000000.0)
            except Exception:
                c_mpa_out[i, j] = 1e-06
    return c_mpa_out

def vdwp_estimate_mixture(temperature, pressure, component_params, composition, structure):
    if pressure <= 0.0001:
        pressure = 0.0001
    r_gas = 8.3144598
    if structure == 1:
        cg_pr_wtr = np.array([1 / 23, 3 / 23])
        del_mu_bulk_0 = 1263.6
        del_vw0 = 4.6e-06
        del_hw0 = -4858.9 if temperature > 273.15 else 1151
    elif structure == 2:
        cg_pr_wtr = np.array([2 / 17, 1 / 17])
        del_mu_bulk_0 = 882.8
        del_vw0 = 5e-06
        del_hw0 = -5202.2 if temperature > 273.15 else 808
    else:
        raise ValueError('Invalid structure')
    fugacities = calculate_mixture_fugacities(pressure, temperature, component_params, np.array(composition))
    langmuir_constants = np.array([kihara_mix(temperature, structure, comp['kihara_params'])[0, :] for comp in component_params])
    theta = np.zeros((2, len(component_params)))
    for j in range(2):
        denominator = 1.0 + np.sum(langmuir_constants[:, j] * fugacities)
        theta[j, :] = langmuir_constants[:, j] * fugacities / max(denominator, 1e-10)
    total_theta = np.clip(np.sum(theta, axis=1), 0.0, 0.999999)
    del_mu_hyd = -r_gas * temperature * np.sum(cg_pr_wtr * np.log(1.0 - total_theta))

    def del_hw_integral_func(t):
        return -38.12 + 0.141 * (t - 273.15)

    def del_hw_integrand(t):
        return (del_hw0 + quad(del_hw_integral_func, 273.15, t)[0]) / (r_gas * t ** 2)
    hw_integral, _ = quad(del_hw_integrand, 273.15, temperature)

    def del_vw_func(p):
        return del_vw0 + 6.695e-12 * p
    vw_integral, _ = quad(lambda p: del_vw_func(p) * 1000000.0 / (r_gas * temperature), 0, pressure)
    del_mu_bulk = (del_mu_bulk_0 / (r_gas * 273.15) - hw_integral + vw_integral - np.log(1.0)) * r_gas * temperature
    return del_mu_bulk - del_mu_hyd

def calculate_multicomponent_equilibrium(temperature, components, composition, structure, params_file='hydrate_parameters.xlsx'):
    if len(components) != len(composition):
        raise ValueError('Components and composition lists must have same length')
    if structure not in [1, 2]:
        raise ValueError('Structure must be 1 (sI) or 2 (sII)')
    composition = [x / sum(composition) for x in composition]
    global _all_bips_df
    if _all_bips_df is None:
        try:
            _all_bips_df = pd.read_excel(params_file, sheet_name='BIPs')
        except Exception:
            _all_bips_df = pd.DataFrame(columns=['Comp1', 'Comp2', 'A_ij', 'B_ij'])
    component_params = [load_component_data(c, params_file) for c in components]
    if temperature < 273.15:
        p_min_init, p_max_init = (0.001, 100.0)
    else:
        p_min_init, p_max_init = (0.01, 300.0)
    if structure == 1:
        p_min, p_max = (p_min_init, p_max_init)
    else:
        p_min, p_max = (p_min_init * 1.2, p_max_init * 1.2)
    log_p_min, log_p_max = (np.log(p_min), np.log(p_max))

    def error_function(log_p):
        try:
            return abs(vdwp_estimate_mixture(temperature, np.exp(log_p), component_params, composition, structure))
        except Exception:
            return 1000000000000.0
    try:
        result = minimize_scalar(error_function, bounds=[log_p_min, log_p_max], method='bounded', options={'xatol': 1e-08, 'maxiter': 500})
        if result.success:
            p_best = np.exp(result.x)
            try:
                residual = abs(vdwp_estimate_mixture(temperature, p_best, component_params, composition, structure))
            except Exception:
                residual = 1000000000000.0
            if residual < 100.0:
                return p_best
            elif residual < 500.0:
                boundary_residual = min(error_function(log_p_min + 1e-06), error_function(log_p_max - 1e-06))
                if residual < boundary_residual * 0.5:
                    return p_best
    except Exception:
        pass
    return None

def predict_structure_and_pressure(temperature, components, composition, params_file='hydrate_parameters.xlsx'):
    try:
        component_params_list = [load_component_data(c, params_file) for c in components]
    except ValueError as e:
        return (None, None, None, None)
    if len(components) == 1:
        preferred_structure = component_params_list[0]['structure']
        pressure = calculate_multicomponent_equilibrium(temperature, components, composition, preferred_structure, params_file)
        sI_pressure = pressure if preferred_structure == 1 else None
        sII_pressure = pressure if preferred_structure == 2 else None
        return (pressure, preferred_structure, sI_pressure, sII_pressure)
    preferred_structures_set = {p['structure'] for p in component_params_list}
    if len(preferred_structures_set) == 1:
        only_structure = preferred_structures_set.pop()
        pressure = calculate_multicomponent_equilibrium(temperature, components, composition, only_structure, params_file)
        sI_pressure = pressure if only_structure == 1 else None
        sII_pressure = pressure if only_structure == 2 else None
        return (pressure, only_structure, sI_pressure, sII_pressure)
    pred_s1 = None
    pred_s2 = None
    try:
        pred_s1 = calculate_multicomponent_equilibrium(temperature, components, composition, 1, params_file)
    except Exception:
        pass
    try:
        pred_s2 = calculate_multicomponent_equilibrium(temperature, components, composition, 2, params_file)
    except Exception:
        pass
    if pred_s1 is None and pred_s2 is None:
        return (None, None, None, None)
    if pred_s1 is not None and (pred_s2 is None or pred_s1 <= pred_s2):
        best_structure = 1
        best_prediction = pred_s1
    else:
        best_structure = 2
        best_prediction = pred_s2
    return (best_prediction, best_structure, pred_s1, pred_s2)

def compute_cage_occupancy(temperature, pressure, component_params, composition, structure):
    fugacities = calculate_mixture_fugacities(pressure, temperature, component_params, np.array(composition))
    langmuir_constants = np.array([kihara_mix(temperature, structure, comp['kihara_params'])[0, :] for comp in component_params])
    theta = np.zeros((2, len(component_params)))
    for j in range(2):
        denominator = 1.0 + np.sum(langmuir_constants[:, j] * fugacities)
        theta[j, :] = langmuir_constants[:, j] * fugacities / max(denominator, 1e-10)
    return (theta, langmuir_constants)

def compute_formation_enthalpy(temperature, components, composition, structure, params_file='hydrate_parameters.xlsx', delta_T=0.5):
    R = 8.3144598
    T1 = temperature - delta_T
    T2 = temperature + delta_T
    P1 = calculate_multicomponent_equilibrium(T1, components, composition, structure, params_file)
    P2 = calculate_multicomponent_equilibrium(T2, components, composition, structure, params_file)
    if P1 is None or P2 is None or P1 <= 0 or (P2 <= 0):
        return None
    d_ln_P = np.log(P2) - np.log(P1)
    d_inv_T = 1.0 / T2 - 1.0 / T1
    delta_H = -R * d_ln_P / d_inv_T
    return delta_H / 1000.0

def compute_water_activity(temperature, inhibitor_type=None, inhibitor_weight_fraction=0.0):
    if inhibitor_type is None or inhibitor_weight_fraction <= 0:
        return 1.0
    inhibitor_params = {'methanol': {'M': 32.04, 'lambda': 1.07}, 'ethylene_glycol': {'M': 62.07, 'lambda': 0.68}, 'NaCl': {'M': 58.44, 'lambda': 1.86, 'nu': 2}, 'CaCl2': {'M': 110.98, 'lambda': 1.5, 'nu': 3}}
    params = inhibitor_params.get(inhibitor_type)
    if params is None:
        return 1.0
    w = inhibitor_weight_fraction
    M_inh = params['M']
    x_inh = w / M_inh / (w / M_inh + (1 - w) / 18.015)
    nu = params.get('nu', 1)
    lam = params['lambda']
    ln_aw = -lam * nu * x_inh / (1 - x_inh + 1e-15)
    a_w = np.clip(np.exp(ln_aw), 0.0, 1.0)
    return a_w

def identify_gas_systems(compositions, component_names):
    systems = []
    for comp in compositions:
        active = tuple(sorted([name for i, name in enumerate(component_names) if comp[i] > 1e-06]))
        systems.append(active)
    unique_systems = sorted(set(systems), key=lambda x: (len(x), x))
    system_map = {s: i for i, s in enumerate(unique_systems)}
    system_labels = np.array([system_map[s] for s in systems])
    return (system_labels, unique_systems, system_map)

def summarize_dataset_composition(temperatures, true_pressures, compositions, component_names, save_csv_path=None):
    tier_label = {1: 'Pure', 2: 'Binary', 3: 'Ternary', 4: 'Quaternary', 5: 'Quinary+'}
    rows = []
    for i in range(len(temperatures)):
        active = tuple(sorted([n for j, n in enumerate(component_names) if compositions[i, j] > 1e-06]))
        rows.append({'System': '+'.join(active) if active else '(empty)', 'n_components': len(active), 'T(K)': float(temperatures[i]), 'p(MPa)': float(true_pressures[i])})
    df = pd.DataFrame(rows)
    total = len(df)
    summary_df = df.groupby(['System', 'n_components']).agg(N_points=('T(K)', 'size'), T_min_K=('T(K)', 'min'), T_max_K=('T(K)', 'max'), P_min_MPa=('p(MPa)', 'min'), P_max_MPa=('p(MPa)', 'max')).reset_index()
    summary_df['Type'] = summary_df['n_components'].clip(upper=5).map(tier_label)
    summary_df['Fraction(%)'] = (summary_df['N_points'] / total * 100).round(2)
    summary_df = summary_df.sort_values(['n_components', 'N_points'], ascending=[True, False]).drop(columns=['n_components']).reset_index(drop=True)
    summary_df = summary_df[['System', 'Type', 'N_points', 'Fraction(%)', 'T_min_K', 'T_max_K', 'P_min_MPa', 'P_max_MPa']]
    total_row = pd.DataFrame([{'System': 'Total', 'Type': '—', 'N_points': total, 'Fraction(%)': 100.0, 'T_min_K': df['T(K)'].min(), 'T_max_K': df['T(K)'].max(), 'P_min_MPa': df['p(MPa)'].min(), 'P_max_MPa': df['p(MPa)'].max()}])
    summary_df = pd.concat([summary_df, total_row], ignore_index=True)
    print('\nDataset composition:')
    print(f'   {'System':<32} {'Type':<10} {'N':>5} {'%':>6}  {'T_min(K)':>9} {'T_max(K)':>9}  {'P_min(MPa)':>11} {'P_max(MPa)':>11}')
    for _, r in summary_df.iterrows():
        print(f'   {r['System']:<32} {r['Type']:<10} {int(r['N_points']):>5} {r['Fraction(%)']:>6.2f}  {r['T_min_K']:>9.2f} {r['T_max_K']:>9.2f}  {r['P_min_MPa']:>11.3f} {r['P_max_MPa']:>11.3f}')
    if save_csv_path:
        try:
            summary_df.to_csv(save_csv_path, index=False, encoding='utf-8-sig')
            print(f"\n 数据集组成统计表已保存: '{save_csv_path}'")
        except Exception as e:
            print(f' 数据集组成统计文件保存失败: {e}')
    return summary_df

def identify_data_sources(dataset):
    source_col_candidates = ['source', 'reference', 'lit_source', 'ref', 'literature', 'Source', 'Reference']
    source_col_used = None
    for col in source_col_candidates:
        if col in dataset.columns:
            source_col_used = col
            break
    if source_col_used is not None:
        raw_labels = dataset[source_col_used].astype(str).values
        unique_sources = sorted(set(raw_labels))
        source_map = {s: i for i, s in enumerate(unique_sources)}
        source_labels = np.array([source_map[s] for s in raw_labels])
        return (source_labels, unique_sources, source_col_used)
    else:
        component_cols = ['xCH4', 'xC2H6', 'xC3H8', 'xCO2', 'xN2', 'xH2S', 'xi-C4H10']
        comp_short = ['CH4', 'C2H6', 'C3H8', 'CO2', 'N2', 'H2S', 'i-C4H10']
        avail_cols = [c for c in component_cols if c in dataset.columns]
        avail_names = [comp_short[component_cols.index(c)] for c in avail_cols]
        systems = []
        for _, row in dataset.iterrows():
            active = tuple(sorted([n for c, n in zip(avail_cols, avail_names) if row.get(c, 0) > 1e-06]))
            systems.append(active)
        unique_sources = sorted(set(systems), key=lambda x: (len(x), x))
        source_map = {s: i for i, s in enumerate(unique_sources)}
        source_labels = np.array([source_map[s] for s in systems])
        return (source_labels, ['+'.join(s) for s in unique_sources], 'gas_system_proxy')
_NON_SPHERICITY_SCORE = {'Methane': 0.0, 'Nitrogen': 0.1, 'Carbon dioxide': 0.5, 'Ethane': 0.5, 'Hydrogen Sulfide': 0.6, 'Propane': 0.8, 'i-Butane': 0.9}

def classify_assumption_violation(temperatures, compositions, component_names, true_pressures):
    n_samples = len(temperatures)
    temperatures = np.asarray(temperatures).flatten()
    compositions = np.atleast_2d(compositions)
    true_pressures = np.asarray(true_pressures).flatten()
    s_nonsphere = np.zeros(n_samples)
    s_multicomp = np.zeros(n_samples)
    s_pressure = np.zeros(n_samples)
    s_double_occ = np.zeros(n_samples)
    for i in range(n_samples):
        comp_row = compositions[i]
        active_mask = comp_row > 1e-06
        if active_mask.any():
            weights = comp_row[active_mask] / comp_row[active_mask].sum()
            scores_i = np.array([_NON_SPHERICITY_SCORE.get(component_names[k], 0.5) for k in range(len(component_names)) if active_mask[k]])
            s_nonsphere[i] = float(np.sum(weights * scores_i))
        n_active = int(active_mask.sum())
        if n_active <= 1:
            s_multicomp[i] = 0.0
        elif n_active == 2:
            s_multicomp[i] = 0.5
        else:
            s_multicomp[i] = 1.0
        p = true_pressures[i]
        if p <= 5.0:
            s_pressure[i] = 0.0
        elif p <= 30.0:
            s_pressure[i] = 0.5 * (p - 5.0) / 25.0
        elif p <= 100.0:
            s_pressure[i] = 0.5 + 0.5 * (p - 30.0) / 70.0
        else:
            s_pressure[i] = 1.0
        small_guest_mask = np.zeros(len(component_names), dtype=bool)
        for k, nm in enumerate(component_names):
            if nm in ('Methane', 'Nitrogen'):
                small_guest_mask[k] = True
        small_guest_frac = float(np.sum(comp_row[small_guest_mask])) if small_guest_mask.any() else 0.0
        if small_guest_frac > 0.5 and p > 50.0:
            s_double_occ[i] = min(1.0, (p - 50.0) / 100.0) * small_guest_frac
    w_nonsphere, w_multicomp, w_pressure, w_double_occ = (0.3, 0.25, 0.3, 0.15)
    scores = w_nonsphere * s_nonsphere + w_multicomp * s_multicomp + w_pressure * s_pressure + w_double_occ * s_double_occ
    group_labels = np.empty(n_samples, dtype=object)
    group_labels[scores <= 0.2] = 'low'
    group_labels[(scores > 0.2) & (scores <= 0.4)] = 'medium'
    group_labels[scores > 0.4] = 'high'
    component_table = pd.DataFrame({'s_nonsphere': s_nonsphere, 's_multicomp': s_multicomp, 's_pressure': s_pressure, 's_double_occ': s_double_occ, 'score': scores, 'group': group_labels})
    return (scores, group_labels, component_table)

def analyze_error_by_assumption_violation(temperatures, compositions, component_names, true_pressures, physics_predictions, hybrid_predictions, save_csv_path='assumption_violation_error_analysis.csv'):
    scores, group_labels, comp_table = classify_assumption_violation(temperatures, compositions, component_names, true_pressures)
    true_p = np.asarray(true_pressures).flatten()
    p_phys = np.asarray(physics_predictions).flatten()
    p_hyb = np.asarray(hybrid_predictions).flatten()
    pct_err_phys = np.abs((p_phys - true_p) / np.maximum(np.abs(true_p), 1e-09)) * 100.0
    pct_err_hyb = np.abs((p_hyb - true_p) / np.maximum(np.abs(true_p), 1e-09)) * 100.0
    per_sample = comp_table.copy()
    per_sample['T(K)'] = temperatures
    per_sample['P_true(MPa)'] = true_p
    per_sample['P_physics(MPa)'] = p_phys
    per_sample['P_hybrid(MPa)'] = p_hyb
    per_sample['pct_err_physics'] = pct_err_phys
    per_sample['pct_err_hybrid'] = pct_err_hyb
    try:
        per_sample.to_csv(save_csv_path, index=False, encoding='utf-8-sig')
        print(f" 分层误差明细已保存: '{save_csv_path}'")
    except Exception as e:
        print(f' CSV 写入失败 (不影响主流程): {e}')
    summary_dict = {}
    table = PrettyTable(['Group', 'N', '%', 'Physics MAPE', 'Hybrid MAPE', 'Physics RMSE', 'Hybrid RMSE', 'Physics R2', 'Hybrid R2'])
    table.align = 'l'
    total_n = len(true_p)
    for grp in ['low', 'medium', 'high']:
        mask = group_labels == grp
        n_g = int(mask.sum())
        if n_g == 0:
            table.add_row([grp, 0, '0.0', '—', '—', '—', '—', '—', '—'])
            summary_dict[grp] = None
            continue
        tp = true_p[mask]
        pp = p_phys[mask]
        ph = p_hyb[mask]
        mape_phys = float(np.mean(np.abs((pp - tp) / np.maximum(np.abs(tp), 1e-09))) * 100.0)
        mape_hyb = float(np.mean(np.abs((ph - tp) / np.maximum(np.abs(tp), 1e-09))) * 100.0)
        rmse_phys = float(np.sqrt(np.mean((pp - tp) ** 2)))
        rmse_hyb = float(np.sqrt(np.mean((ph - tp) ** 2)))
        if n_g >= 3:
            try:
                r2_phys = float(r2_score(tp, pp))
            except Exception:
                r2_phys = float('nan')
            try:
                r2_hyb = float(r2_score(tp, ph))
            except Exception:
                r2_hyb = float('nan')
        else:
            r2_phys = float('nan')
            r2_hyb = float('nan')
        summary_dict[grp] = {'n': n_g, 'mape_physics': mape_phys, 'mape_hybrid': mape_hyb, 'rmse_physics': rmse_phys, 'rmse_hybrid': rmse_hyb, 'r2_physics': r2_phys, 'r2_hybrid': r2_hyb}
        table.add_row([grp, n_g, f'{n_g / max(total_n, 1) * 100:.1f}', f'{mape_phys:.2f}', f'{mape_hyb:.2f}', f'{rmse_phys:.4f}', f'{rmse_hyb:.4f}', f'{r2_phys:.4f}' if not np.isnan(r2_phys) else '—', f'{r2_hyb:.4f}' if not np.isnan(r2_hyb) else '—'])
    print('\nAssumption-violation stratified errors:')
    print(table)
    return (summary_dict, per_sample)

def polynomial_correction_baseline(temperatures_train, compositions_train, physics_pred_train, y_train, temperatures_test, compositions_test, physics_pred_test, y_test, true_pressures_test, physics_pressures_test):
    X_poly_train = np.column_stack([1.0 / temperatures_train, compositions_train, np.log(np.maximum(physics_pred_train, 1e-09))])
    X_poly_test = np.column_stack([1.0 / temperatures_test, compositions_test, np.log(np.maximum(physics_pred_test, 1e-09))])
    poly_pipeline = Pipeline([('poly', PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)), ('ridge', Ridge(alpha=1.0))])
    poly_pipeline.fit(X_poly_train, y_train)
    poly_pred_residuals = poly_pipeline.predict(X_poly_test)
    log_physics_test = np.log(np.maximum(physics_pressures_test, 1e-09))
    poly_predictions = np.exp(log_physics_test + poly_pred_residuals)
    n_poly_features = poly_pipeline.named_steps['poly'].n_output_features_
    print(f'\nPolynomial baseline (features={n_poly_features}):')
    poly_metrics = evaluate_forecasts(true_pressures_test, poly_predictions, '多项式基线修正')
    return (poly_pipeline, poly_metrics, poly_predictions)

def margules_correction_baseline(temperatures_train, compositions_train, physics_pred_train, y_train, temperatures_test, compositions_test, physics_pred_test, y_test, true_pressures_test, physics_pressures_test):
    n_comp = compositions_train.shape[1]

    def _build(temps, comps, phys_pred):
        feat_list = [1.0 / temps, *[comps[:, i] for i in range(n_comp)], *[comps[:, i] * comps[:, j] for i in range(n_comp) for j in range(i + 1, n_comp)], np.log(np.maximum(temps, 1e-09)), np.log(np.maximum(phys_pred, 1e-09))]
        return np.column_stack(feat_list)
    X_m_train = _build(temperatures_train, compositions_train, physics_pred_train)
    X_m_test = _build(temperatures_test, compositions_test, physics_pred_test)
    margules_pipeline = Pipeline([('ridge', Ridge(alpha=1.0))])
    margules_pipeline.fit(X_m_train, y_train)
    margules_pred_residuals = margules_pipeline.predict(X_m_test)
    log_physics_test = np.log(np.maximum(physics_pressures_test, 1e-09))
    margules_predictions = np.exp(log_physics_test + margules_pred_residuals)
    n_margules_features = X_m_train.shape[1]
    n_margules_params = n_margules_features + 1
    print(f'\nMargules baseline (features={n_margules_features}):')
    margules_metrics = evaluate_forecasts(true_pressures_test, margules_predictions, 'Margules型修正')
    return (margules_pipeline, margules_metrics, margules_predictions)

def rmsle(y_true, y_pred):
    return np.sqrt(mean_squared_error(np.log1p(y_true), np.log1p(y_pred)))

def mape(y_true, y_pred):
    y_true, y_pred = (np.array(y_true), np.array(y_pred))
    return np.mean(np.abs((y_true - y_pred) / (y_true + 1e-08))) * 100

def evaluate_forecasts(y_true, y_pred, model_name):
    table = PrettyTable(['模型', '指标', '值', '单位'])
    table.align = 'l'
    actual, predicted = (y_true.flatten(), y_pred.flatten())
    p_range = actual.max() - actual.min()
    p_median = np.median(actual)
    metrics = {'MSE': mean_squared_error(actual, predicted), 'RMSE': sqrt(mean_squared_error(actual, predicted)), 'MAE': mean_absolute_error(actual, predicted), 'R²': r2_score(actual, predicted), 'MAPE': mape(actual, predicted), 'RMSLE': rmsle(actual, predicted), 'Max Error': max_error(actual, predicted), 'Max Error / Range': max_error(actual, predicted) / max(p_range, 1e-09) * 100, 'Max Error / Median': max_error(actual, predicted) / max(p_median, 1e-09) * 100}
    table.add_row([model_name, 'MSE', f'{metrics['MSE']:.4f}', 'MPa²'])
    table.add_row([model_name, 'RMSE', f'{metrics['RMSE']:.4f}', 'MPa'])
    table.add_row([model_name, 'MAE', f'{metrics['MAE']:.4f}', 'MPa'])
    table.add_row([model_name, 'R²', f'{metrics['R²']:.4f}', '[-]'])
    table.add_row([model_name, 'MAPE', f'{metrics['MAPE']:.2f}', '%'])
    table.add_row([model_name, 'RMSLE', f'{metrics['RMSLE']:.4f}', '[-]'])
    table.add_row([model_name, 'Max Error', f'{metrics['Max Error']:.4f}', 'MPa'])
    table.add_row([model_name, 'Max Error/Range', f'{metrics['Max Error / Range']:.2f}', '% of test range'])
    table.add_row([model_name, 'Max Error/Median', f'{metrics['Max Error / Median']:.2f}', '% of median P'])
    table.add_row([model_name, '测试集压力范围', f'[{actual.min():.3f}, {actual.max():.3f}]', 'MPa'])
    print(table)
    return metrics

def create_enhanced_features(temperatures, compositions, physics_predictions, structures):
    features = []
    temperatures = np.array(temperatures).flatten()
    compositions = np.atleast_2d(compositions)
    physics_predictions = np.array(physics_predictions).flatten()
    structures = np.array(structures).flatten()
    features.append(temperatures)
    features.extend([compositions[:, i] for i in range(compositions.shape[1])])
    features.append(np.log(np.maximum(physics_predictions, 1e-09)))
    features.append(structures)
    features.append(temperatures ** 2)
    features.append(1 / (temperatures + 1e-09))
    features.append(np.sqrt(temperatures))
    total_heavy = np.sum(compositions[:, [1, 2, 6]], axis=1)
    features.append(total_heavy)
    methane_ratio = compositions[:, 0] / (np.sum(compositions[:, 1:], axis=1) + 1e-09)
    features.append(methane_ratio)
    co2_conc = compositions[:, 3]
    features.append(co2_conc * temperatures)
    features.append(physics_predictions ** 0.5)
    features.append(physics_predictions ** 2)
    return np.column_stack(features)

def fix_clustering_for_stratification(error_clusters, min_samples_per_cluster=2):
    cluster_counts = Counter(error_clusters)
    insufficient_clusters = [c for c, count in cluster_counts.items() if count < min_samples_per_cluster]
    if not insufficient_clusters:
        return error_clusters
    largest_cluster = max(cluster_counts, key=cluster_counts.get)
    fixed_clusters = error_clusters.copy()
    for ic in insufficient_clusters:
        fixed_clusters[fixed_clusters == ic] = largest_cluster
    unique_clusters = np.unique(fixed_clusters)
    mapping = {old: new for new, old in enumerate(unique_clusters)}
    return np.array([mapping[c] for c in fixed_clusters])

def _convert_to_jsonable(value):
    if isinstance(value, dict):
        return {str(k): _convert_to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_convert_to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value

def export_supplementary_model_files(lgb_model, rf_model, scaler_X, model_config, params_file='hydrate_parameters.xlsx', output_dir='supplementary_model_files'):
    os.makedirs(output_dir, exist_ok=True)
    architecture_summary = {'model_name': 'vdW-P + residual-learning hybrid hydrate equilibrium model', 'prediction_equation': 'P_hybrid = exp(log(P_vdW-P) + 0.65*r_LightGBM + 0.35*r_RandomForest)', 'residual_target': 'r = log(P_exp) - log(P_vdW-P)', 'input_features': model_config.get('feature_names', []), 'component_order': model_config.get('component_order', []), 'ensemble_weights': model_config.get('ensemble_weights', []), 'log_floor': model_config.get('log_floor', 1e-09), 'lightgbm_best_params': model_config.get('best_lgb_params', {}), 'random_forest_params': rf_model.get_params(), 'scaler_mean': getattr(scaler_X, 'mean_', np.array([])), 'scaler_scale': getattr(scaler_X, 'scale_', np.array([])), 'scaler_var': getattr(scaler_X, 'var_', np.array([])), 'training_temp_range': model_config.get('training_temp_range', None), 'training_pressure_range': model_config.get('training_pressure_range', None), 'model_version': model_config.get('model_version', None), 'training_date': model_config.get('training_date', None), 'software': {'python': sys.version, 'platform': platform.platform(), 'lightgbm': getattr(lgb, '__version__', 'unknown')}}
    with open(os.path.join(output_dir, 'model_architecture_and_parameters.json'), 'w', encoding='utf-8') as f:
        json.dump(_convert_to_jsonable(architecture_summary), f, indent=2, ensure_ascii=False)
    feature_table = pd.DataFrame({'feature_name': model_config.get('feature_names', []), 'standard_scaler_mean': getattr(scaler_X, 'mean_', np.array([])), 'standard_scaler_scale': getattr(scaler_X, 'scale_', np.array([])), 'standard_scaler_variance': getattr(scaler_X, 'var_', np.array([]))})
    feature_table.to_csv(os.path.join(output_dir, 'feature_scaling_parameters.csv'), index=False, encoding='utf-8-sig')
    if hasattr(lgb_model, 'booster_'):
        lgb_model.booster_.save_model(os.path.join(output_dir, 'lightgbm_tree_model.txt'))
        with open(os.path.join(output_dir, 'lightgbm_tree_model.json'), 'w', encoding='utf-8') as f:
            json.dump(_convert_to_jsonable(lgb_model.booster_.dump_model()), f, indent=2, ensure_ascii=False)
    rf_trees = []
    for tree_idx, estimator in enumerate(rf_model.estimators_):
        tree = estimator.tree_
        rf_trees.append({'tree_index': tree_idx, 'n_nodes': tree.node_count, 'children_left': tree.children_left, 'children_right': tree.children_right, 'feature': tree.feature, 'threshold': tree.threshold, 'value': tree.value.squeeze(), 'impurity': tree.impurity, 'n_node_samples': tree.n_node_samples, 'weighted_n_node_samples': tree.weighted_n_node_samples})
    with open(os.path.join(output_dir, 'random_forest_tree_structures.json'), 'w', encoding='utf-8') as f:
        json.dump(_convert_to_jsonable(rf_trees), f, indent=2, ensure_ascii=False)
    readme_text = 'Supplementary model files\n=========================\n\nThis folder contains all fitted parameters required to reproduce the hybrid model:\n1. model_architecture_and_parameters.json: model equation, ensemble weights, hyperparameters, feature list, scaler parameters, and software versions.\n2. feature_scaling_parameters.csv: feature order and StandardScaler mean/scale/variance.\n3. lightgbm_tree_model.txt/json: complete LightGBM tree structure and leaf values.\n4. random_forest_tree_structures.json: complete Random Forest split features, thresholds, children, and leaf values for every tree.\n5. parameter_sheet_*.csv: thermodynamic parameters used by the vdW-P model.\n\nBecause LightGBM and Random Forest are tree-ensemble models, their fitted quantities are tree split thresholds and leaf values rather than linear coefficients.\n'
    with open(os.path.join(output_dir, 'README_model_reproducibility.txt'), 'w', encoding='utf-8') as f:
        f.write(readme_text)
    if os.path.exists(params_file):
        try:
            parameter_sheets = pd.read_excel(params_file, sheet_name=None)
            for sheet_name, sheet_df in parameter_sheets.items():
                safe_sheet_name = ''.join((ch if ch.isalnum() or ch in ('-', '_') else '_' for ch in str(sheet_name)))
                sheet_df.to_csv(os.path.join(output_dir, f'parameter_sheet_{safe_sheet_name}.csv'), index=False, encoding='utf-8-sig')
        except Exception as e:
            print(f' 物理参数表导出失败: {e}')
    print(f" 补充材料模型文件已导出至: '{output_dir}'")

def train_and_save_models():
    print('Hydrate phase-equilibrium model training')
    print('[1/8] Loading data...')
    params_file = 'hydrate_parameters.xlsx'
    if not os.path.exists(params_file):
        print(' 参数文件不存在，正在创建...')
        create_parameter_template(params_file)
    data_filename = 'date.csv'
    try:
        dataset = pd.read_csv(data_filename, encoding='utf-8-sig').dropna()
        if 'p(MPa)' not in dataset.columns and 'p_true(MPa)' in dataset.columns:
            dataset.rename(columns={'p_true(MPa)': 'p(MPa)'}, inplace=True)
        print(f" 成功加载数据集: '{data_filename}'")
        print(f' 有效样本数量: {len(dataset)} 条')
        print(f' 数据维度: {dataset.shape}')
    except Exception as e:
        print(f' 数据加载失败: {e}')
        return
    min_temp_train, max_temp_train = (dataset['T(K)'].min(), dataset['T(K)'].max())
    print(f' 训练数据统计:')
    print(f' • 温度范围: [{min_temp_train:.2f}K, {max_temp_train:.2f}K]')
    print(f' • 压力范围: [{dataset['p(MPa)'].min():.3f}, {dataset['p(MPa)'].max():.3f}] MPa')
    original_count = len(dataset)
    filter_log = []
    P_MAX_ENGINEERING = 200.0
    P_MIN_ENGINEERING = 0.01
    mask_pressure = (dataset['p(MPa)'] >= P_MIN_ENGINEERING) & (dataset['p(MPa)'] <= P_MAX_ENGINEERING)
    n_pressure_removed = (~mask_pressure).sum()
    if n_pressure_removed > 0:
        filter_log.append(f'   • 压力越界: 移除 {n_pressure_removed} 个样本 (范围外 [{P_MIN_ENGINEERING}, {P_MAX_ENGINEERING}] MPa)')
    dataset = dataset[mask_pressure]
    T_MIN_HYDRATE = 190.0
    T_MAX_HYDRATE = 340.0
    mask_temp = (dataset['T(K)'] >= T_MIN_HYDRATE) & (dataset['T(K)'] <= T_MAX_HYDRATE)
    n_temp_removed = (~mask_temp).sum()
    if n_temp_removed > 0:
        filter_log.append(f'   • 温度越界: 移除 {n_temp_removed} 个样本 (范围外 [{T_MIN_HYDRATE}, {T_MAX_HYDRATE}] K)')
    dataset = dataset[mask_temp]
    component_cols_check = ['xCH4', 'xC2H6', 'xC3H8', 'xCO2', 'xN2', 'xH2S', 'xi-C4H10']
    comp_sum = dataset[component_cols_check].sum(axis=1)
    mask_comp = (comp_sum >= 0.95) & (comp_sum <= 1.05)
    n_comp_removed = (~mask_comp).sum()
    if n_comp_removed > 0:
        filter_log.append(f'   • 组分摩尔分数之和偏差 >5%: 移除 {n_comp_removed} 个样本 (sum范围: [{comp_sum.min():.3f}, {comp_sum.max():.3f}])')
    dataset = dataset[mask_comp]
    total_filtered = original_count - len(dataset)
    if filter_log:
        print(f'\n 多重数据质量过滤 (共移除 {total_filtered} 个样本, 占原始数据 {total_filtered / original_count * 100:.1f}%):')
        for log_entry in filter_log:
            print(log_entry)
        print(f'   • 过滤后样本数: {len(dataset)} 条')
        print(f'   • 过滤后压力范围: [{dataset['p(MPa)'].min():.3f}, {dataset['p(MPa)'].max():.3f}] MPa')
        print(f'   • 过滤后温度范围: [{dataset['T(K)'].min():.2f}, {dataset['T(K)'].max():.2f}] K')
    else:
        print(f' 所有 {original_count} 个样本均通过多重数据质量检验')
    component_cols = ['xCH4', 'xC2H6', 'xC3H8', 'xCO2', 'xN2', 'xH2S', 'xi-C4H10']
    component_names = ['Methane', 'Ethane', 'Propane', 'Carbon dioxide', 'Nitrogen', 'Hydrogen Sulfide', 'i-Butane']
    temperatures = dataset['T(K)'].values
    true_pressures = dataset['p(MPa)'].values
    compositions = dataset[component_cols].values
    print('\n[2/8] Physics model batch calculation...')
    physics_predictions, selected_structures, valid_indices = ([], [], [])
    failed_count = 0
    for idx in tqdm(range(len(temperatures)), desc='Physics'):
        temp, comp = (temperatures[idx], compositions[idx])
        active_comps = [name for i, name in enumerate(component_names) if comp[i] > 1e-06]
        active_comp_fracs = [c for c in comp if c > 1e-06]
        if not active_comps:
            failed_count += 1
            continue
        try:
            best_pred, best_struct, _, _ = predict_structure_and_pressure(temp, active_comps, active_comp_fracs, params_file=params_file)
            if best_pred is not None:
                physics_predictions.append(best_pred)
                selected_structures.append(best_struct)
                valid_indices.append(idx)
            else:
                failed_count += 1
        except Exception:
            failed_count += 1
    print(f' 物理模型计算完成')
    print(f' • 成功计算: {len(valid_indices)} 个样本')
    print(f' • 计算失败: {failed_count} 个样本')
    print(f' • 成功率: {len(valid_indices) / (len(valid_indices) + failed_count) * 100:.1f}%')
    if not valid_indices:
        print(' 没有任何样本通过物理模型计算，无法继续训练')
        return
    structure_counts = Counter(selected_structures)
    print(f' 晶体结构分布分析:')
    print(f' • 结构 I (sI): {structure_counts.get(1, 0)} 次 ({structure_counts.get(1, 0) / len(selected_structures) * 100:.1f}%)')
    print(f' • 结构 II (sII): {structure_counts.get(2, 0)} 次 ({structure_counts.get(2, 0) / len(selected_structures) * 100:.1f}%)')
    valid_temperatures = temperatures[valid_indices]
    valid_true_pressures = true_pressures[valid_indices]
    valid_compositions = compositions[valid_indices]
    physics_predictions = np.array(physics_predictions)
    selected_structures = np.array(selected_structures)
    print('\n[3/8] Feature engineering...')
    LOG_FLOOR = 1e-09
    log_residuals = np.log(np.maximum(valid_true_pressures, LOG_FLOOR)) - np.log(np.maximum(physics_predictions, LOG_FLOOR))
    print(f' 对数残差统计 (过滤前):')
    print(f' • 平均值: {np.mean(log_residuals):.4f}')
    print(f' • 标准差: {np.std(log_residuals):.4f}')
    print(f' • 范围: [{np.min(log_residuals):.4f}, {np.max(log_residuals):.4f}]')
    print('\n[3.5] Outlier removal (3x IQR on log residuals)...')
    q1_res = np.percentile(log_residuals, 25)
    q3_res = np.percentile(log_residuals, 75)
    iqr_res = q3_res - q1_res
    lb_res = q1_res - 3.0 * iqr_res
    ub_res = q3_res + 3.0 * iqr_res
    inlier_mask = (log_residuals >= lb_res) & (log_residuals <= ub_res)
    n_iqr_removed = int(np.sum(~inlier_mask))
    print(f'   • 对数残差 IQR = {iqr_res:.4f}')
    print(f'   • 3×IQR 过滤窗口: [{lb_res:.4f}, {ub_res:.4f}]')
    if n_iqr_removed > 0:
        print(f'   Removed {n_iqr_removed} outliers ({n_iqr_removed / len(log_residuals) * 100:.1f}%)')
        valid_temperatures = valid_temperatures[inlier_mask]
        valid_true_pressures = valid_true_pressures[inlier_mask]
        valid_compositions = valid_compositions[inlier_mask]
        physics_predictions = physics_predictions[inlier_mask]
        selected_structures = selected_structures[inlier_mask]
        log_residuals = log_residuals[inlier_mask]
        print(f'   • 过滤后有效样本数: {len(log_residuals)}')
        print(f' 对数残差统计 (过滤后):')
        print(f' • 平均值: {np.mean(log_residuals):.4f}')
        print(f' • 标准差: {np.std(log_residuals):.4f}')
        print(f' • 范围: [{np.min(log_residuals):.4f}, {np.max(log_residuals):.4f}]')
    else:
        print(f'    未检测到极端异常点（所有对数残差均在 3-IQR 范围内）')
    feature_names = ['temperature', 'comp_ch4', 'comp_c2h6', 'comp_c3h8', 'comp_co2', 'comp_n2', 'comp_h2s', 'comp_ic4h10', 'log_physics_pred', 'structure', 'temp_sq', 'temp_inv', 'temp_sqrt', 'total_heavy', 'methane_ratio', 'co2_x_temp', 'pressure_sqrt', 'pressure_sq']
    X_features_np = create_enhanced_features(valid_temperatures, valid_compositions, physics_predictions, selected_structures)
    X_features = pd.DataFrame(X_features_np, columns=feature_names)
    y_target = log_residuals
    print(f' 特征工程完成:')
    print(f' • 特征数量: {X_features.shape[1]} 个')
    print(f' • 样本数量: {X_features.shape[0]} 个')
    comp_short_names = ['CH4', 'C2H6', 'C3H8', 'CO2', 'N2', 'H2S', 'i-C4H10']
    system_labels, unique_systems, system_map = identify_gas_systems(valid_compositions, comp_short_names)
    print(f'\n 气体体系识别 (共 {len(unique_systems)} 个独立体系，用于 Leave-One-System-Out CV):')
    for sys_name, sys_id in system_map.items():
        count = np.sum(system_labels == sys_id)
        print(f'   • {'+'.join(sys_name)}: {count} 个样本')
    summarize_dataset_composition(temperatures=valid_temperatures, true_pressures=valid_true_pressures, compositions=valid_compositions, component_names=comp_short_names, save_csv_path='dataset_composition_summary.csv')
    valid_dataset_slice = dataset.iloc[valid_indices].reset_index(drop=True)
    if n_iqr_removed > 0:
        valid_dataset_slice = valid_dataset_slice[inlier_mask.values if hasattr(inlier_mask, 'values') else inlier_mask].reset_index(drop=True)
    source_labels_full, unique_sources, source_col_used = identify_data_sources(valid_dataset_slice)
    print(f"\n 文献来源识别 (列='{source_col_used}', 共 {len(unique_sources)} 个来源，用于 Leave-One-Source-Out CV):")
    for i, src in enumerate(unique_sources):
        count = int(np.sum(source_labels_full == i))
        print(f'   • {src}: {count} 个样本')
    print('\n[4/10] Train/test split...')
    n_clusters = min(4, len(X_features) // 5)
    stratify_col = None
    if n_clusters >= 2:
        print(f' 使用 {n_clusters} 个聚类进行分层抽样...')
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
        error_clusters = kmeans.fit_predict(log_residuals.reshape(-1, 1))
        stratify_col = fix_clustering_for_stratification(error_clusters)
        print(f' 分层抽样设置完成')
    else:
        print(' 样本数较少，使用普通随机抽样')
    X_train, X_test, y_train, y_test, train_indices, test_indices = train_test_split(X_features, y_target, np.arange(len(X_features)), test_size=0.2, random_state=42, stratify=stratify_col)
    print(f' 数据划分完成:')
    print(f' • 训练集: {len(X_train)} 条样本 ({len(X_train) / len(X_features) * 100:.1f}%)')
    print(f' • 测试集: {len(X_test)} 条样本 ({len(X_test) / len(X_features) * 100:.1f}%)')
    print(' 设计样本权重...')
    true_train_pressure = valid_true_pressures[train_indices]
    base_weight = 1.0
    error_weight = np.abs(y_train) * 2.0
    low_pressure_bonus = (true_train_pressure < 2.0) * 2.0
    sample_weights = base_weight + error_weight + low_pressure_bonus
    print(' 执行特征标准化...')
    scaler_X = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler_X.fit_transform(X_train), columns=X_train.columns)
    X_test_scaled = pd.DataFrame(scaler_X.transform(X_test), columns=X_test.columns)
    print(' 特征标准化完成')
    print('\n[5/10] Group CV (LOSO)...')

    def _run_group_cv(group_labels, group_names, label_prefix, min_groups=3):
        n_unique = len(np.unique(group_labels))
        if n_unique < min_groups:
            print(f'    [{label_prefix}] 分组数不足({n_unique}<{min_groups})，跳过')
            return (None, None)
        n_splits = min(n_unique, 5)
        gkf = GroupKFold(n_splits=n_splits)
        cv_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
        r2_list, mape_list = ([], [])
        print(f'\n   [{label_prefix}] n_splits={n_splits}, 分组数={n_unique}')
        for fold_idx, (cv_tr, cv_te) in enumerate(gkf.split(X_features, y_target, groups=group_labels)):
            sc = StandardScaler()
            X_tr_ = sc.fit_transform(X_features.iloc[cv_tr])
            X_te_ = sc.transform(X_features.iloc[cv_te])
            cv_model.fit(X_tr_, y_target[cv_tr])
            pred_ = cv_model.predict(X_te_)
            log_p_ = np.log(np.maximum(physics_predictions[cv_te], LOG_FLOOR))
            hyb_ = np.exp(log_p_ + pred_)
            true_ = valid_true_pressures[cv_te]
            r2_ = r2_score(true_, hyb_)
            mape_ = mape(true_, hyb_)
            r2_list.append(r2_)
            mape_list.append(mape_)
            held_out_ids = set(group_labels[cv_te])
            held_names = [str(group_names[i]) if i < len(group_names) else str(i) for i in held_out_ids]
            print(f'     Fold {fold_idx + 1}: R²={r2_:.4f}, MAPE={mape_:.2f}%  |  保留: {held_names}')
        r2_arr = np.array(r2_list)
        mape_arr = np.array(mape_list)
        print(f'\n   [{label_prefix}] 汇总:  mean R²={r2_arr.mean():.4f}±{r2_arr.std():.4f}, min R²={r2_arr.min():.4f}, mean MAPE={mape_arr.mean():.2f}%')
        return (r2_arr, mape_arr)
    sys_names_flat = ['+'.join(s) for s in unique_systems]
    r2_sys, mape_sys = _run_group_cv(system_labels, sys_names_flat, 'LOSO-System')
    r2_src, mape_src = _run_group_cv(source_labels_full, unique_sources, 'LOSO-Source')
    if r2_sys is not None:
        print(f'LOSO-System min R²={r2_sys.min():.4f}')
    if r2_src is not None:
        print(f'LOSO-Source min R²={r2_src.min():.4f}')
    print('\n[6/10] LightGBM training...')
    lgbm = lgb.LGBMRegressor(random_state=42, verbose=-1)
    param_grid = {'n_estimators': [200, 400, 600], 'learning_rate': [0.05, 0.1, 0.15], 'num_leaves': [31, 50, 70], 'max_depth': [-1, 15, 25], 'min_samples_split': [10, 20], 'subsample': [0.8, 1.0]}
    gscv = GridSearchCV(estimator=lgbm, param_grid=param_grid, cv=3, n_jobs=-1, scoring='neg_mean_squared_error', verbose=1)
    gscv.fit(X_train_scaled, y_train, sample_weight=sample_weights)
    lgb_model = gscv.best_estimator_
    print('LightGBM done:', gscv.best_params_)
    print('\n[7/10] Random Forest training...')
    rf_model = RandomForestRegressor(random_state=42, n_estimators=300, max_depth=15, min_samples_split=5, min_samples_leaf=2, max_features='sqrt', n_jobs=-1)
    rf_model.fit(X_train_scaled, y_train, sample_weight=sample_weights)
    print('Random Forest done')
    print('\n[8/10] Test-set evaluation...')
    lgb_pred_residuals = lgb_model.predict(X_test_scaled)
    rf_pred_residuals = rf_model.predict(X_test_scaled)
    ensemble_weights = [0.65, 0.35]
    predicted_log_residuals = ensemble_weights[0] * lgb_pred_residuals + ensemble_weights[1] * rf_pred_residuals
    physics_test_pressure = physics_predictions[test_indices]
    log_physics_test = np.log(np.maximum(physics_test_pressure, LOG_FLOOR))
    hybrid_predictions = np.exp(log_physics_test + predicted_log_residuals)
    true_test_pressure = valid_true_pressures[test_indices]
    physics_metrics = evaluate_forecasts(true_test_pressure, physics_test_pressure, 'Physics')
    hybrid_metrics = evaluate_forecasts(true_test_pressure, hybrid_predictions, 'Hybrid')
    analyze_error_by_assumption_violation(valid_temperatures[test_indices], valid_compositions[test_indices], component_names, true_test_pressure, physics_test_pressure, hybrid_predictions)
    print('\n[9/10] Baseline comparison (polynomial & Margules)...')
    poly_pipeline, poly_metrics, poly_predictions = polynomial_correction_baseline(valid_temperatures[train_indices], valid_compositions[train_indices], physics_predictions[train_indices], y_train, valid_temperatures[test_indices], valid_compositions[test_indices], physics_predictions[test_indices], y_test, true_test_pressure, physics_test_pressure)
    margules_pipeline, margules_metrics, margules_predictions = margules_correction_baseline(valid_temperatures[train_indices], valid_compositions[train_indices], physics_predictions[train_indices], y_train, valid_temperatures[test_indices], valid_compositions[test_indices], physics_predictions[test_indices], y_test, true_test_pressure, physics_test_pressure)
    test_sys_labels = system_labels[test_indices]
    train_sys_labels = system_labels[train_indices]
    interpolation_mask = np.isin(test_sys_labels, train_sys_labels)
    n_interp = int(interpolation_mask.sum())
    n_extrap = int((~interpolation_mask).sum())
    print(f'Interpolation: {n_interp}, extrapolation: {n_extrap}')
    if n_interp > 0 and n_extrap > 0:
        interp_r2 = r2_score(true_test_pressure[interpolation_mask], hybrid_predictions[interpolation_mask])
        extrap_r2 = r2_score(true_test_pressure[~interpolation_mask], hybrid_predictions[~interpolation_mask])
        interp_mape = mape(true_test_pressure[interpolation_mask], hybrid_predictions[interpolation_mask])
        extrap_mape = mape(true_test_pressure[~interpolation_mask], hybrid_predictions[~interpolation_mask])
        print(f'  Interp: R²={interp_r2:.4f}, MAPE={interp_mape:.2f}%')
        print(f'  Extrap: R²={extrap_r2:.4f}, MAPE={extrap_mape:.2f}%')
    p_med_test = np.median(true_test_pressure)
    comparison_table = PrettyTable(['模型', '参数量', 'R²', 'RMSE (MPa)', 'MAPE (%)', 'Max Error (MPa)'])
    comparison_table.align = 'l'
    comparison_table.add_row(['vdW-P', '—', f'{physics_metrics['R²']:.4f}', f'{physics_metrics['RMSE']:.4f}', f'{physics_metrics['MAPE']:.2f}', f'{physics_metrics['Max Error']:.4f}'])
    comparison_table.add_row(['Polynomial baseline', f'~{poly_pipeline.named_steps['poly'].n_output_features_}', f'{poly_metrics['R²']:.4f}', f'{poly_metrics['RMSE']:.4f}', f'{poly_metrics['MAPE']:.2f}', f'{poly_metrics['Max Error']:.4f}'])
    comparison_table.add_row(['Margules baseline', f'~{7 + 7 * 6 // 2 + 3}', f'{margules_metrics['R²']:.4f}', f'{margules_metrics['RMSE']:.4f}', f'{margules_metrics['MAPE']:.2f}', f'{margules_metrics['Max Error']:.4f}'])
    comparison_table.add_row(['Hybrid (LGB+RF)', '—', f'{hybrid_metrics['R²']:.4f}', f'{hybrid_metrics['RMSE']:.4f}', f'{hybrid_metrics['MAPE']:.2f}', f'{hybrid_metrics['Max Error']:.4f}'])
    print(comparison_table)
    print('\nThermodynamic consistency check (sample)...')
    n_thermo_samples = min(5, len(test_indices))
    sample_idx_for_thermo = np.random.RandomState(42).choice(len(test_indices), n_thermo_samples, replace=False)
    thermo_table = PrettyTable(['样本', 'T(K)', 'P_true', 'P_hybrid', '结构', 'ΔH(kJ/mol)', 'θ_小笼', 'θ_大笼'])
    thermo_table.align = 'l'
    for si in sample_idx_for_thermo:
        idx_in_valid = test_indices[si]
        t_sample = valid_temperatures[idx_in_valid]
        p_hybrid_sample = hybrid_predictions[si]
        struct_sample = int(selected_structures[idx_in_valid])
        comp_sample = valid_compositions[idx_in_valid]
        active_names = [component_names[k] for k in range(len(component_names)) if comp_sample[k] > 1e-06]
        active_fracs_raw = [comp_sample[k] for k in range(len(comp_sample)) if comp_sample[k] > 1e-06]
        total_frac = sum(active_fracs_raw)
        active_fracs = [f / total_frac for f in active_fracs_raw]
        delta_h_str = 'N/A'
        try:
            delta_h = compute_formation_enthalpy(t_sample, active_names, active_fracs, struct_sample, params_file)
            if delta_h is not None:
                delta_h_str = f'{delta_h:.2f}'
        except Exception:
            pass
        theta_small_str = 'N/A'
        theta_large_str = 'N/A'
        try:
            comp_params_sample = [load_component_data(c, params_file) for c in active_names]
            theta, _ = compute_cage_occupancy(t_sample, p_hybrid_sample, comp_params_sample, active_fracs, struct_sample)
            theta_small_str = f'{np.sum(theta[0, :]):.4f}'
            theta_large_str = f'{np.sum(theta[1, :]):.4f}'
        except Exception:
            pass
        thermo_table.add_row([si, f'{t_sample:.2f}', f'{valid_true_pressures[idx_in_valid]:.3f}', f'{p_hybrid_sample:.3f}', 'sI' if struct_sample == 1 else 'sII', delta_h_str, theta_small_str, theta_large_str])
    print(thermo_table)
    print('Thermodynamic statistics (test subset):')
    n_thermo_quant = min(120, len(test_indices))
    quant_indices = np.arange(len(test_indices))
    if len(quant_indices) > n_thermo_quant:
        quant_indices = np.random.RandomState(123).choice(len(test_indices), n_thermo_quant, replace=False)
    delta_h_values = []
    delta_h_nonpositive_count = 0
    theta_small_values = []
    theta_large_values = []
    theta_bound_violations = 0
    theta_eval_count = 0
    for si in quant_indices:
        idx_in_valid = test_indices[si]
        t_sample = valid_temperatures[idx_in_valid]
        p_hybrid_sample = hybrid_predictions[si]
        struct_sample = int(selected_structures[idx_in_valid])
        comp_sample = valid_compositions[idx_in_valid]
        active_names = [component_names[k] for k in range(len(component_names)) if comp_sample[k] > 1e-06]
        active_fracs_raw = [comp_sample[k] for k in range(len(comp_sample)) if comp_sample[k] > 1e-06]
        total_frac = sum(active_fracs_raw)
        active_fracs = [f / total_frac for f in active_fracs_raw]
        try:
            delta_h = compute_formation_enthalpy(t_sample, active_names, active_fracs, struct_sample, params_file)
            if delta_h is not None and np.isfinite(delta_h):
                delta_h_values.append(delta_h)
                if delta_h <= 0:
                    delta_h_nonpositive_count += 1
        except Exception:
            pass
        try:
            comp_params_sample = [load_component_data(c, params_file) for c in active_names]
            theta, _ = compute_cage_occupancy(t_sample, p_hybrid_sample, comp_params_sample, active_fracs, struct_sample)
            theta_small = float(np.sum(theta[0, :]))
            theta_large = float(np.sum(theta[1, :]))
            theta_small_values.append(theta_small)
            theta_large_values.append(theta_large)
            theta_eval_count += 1
            tol = 1e-06
            if theta_small < -tol or theta_small > 1.0 + tol:
                theta_bound_violations += 1
            if theta_large < -tol or theta_large > 1.0 + tol:
                theta_bound_violations += 1
        except Exception:
            pass
    positive_pressure_rate = np.mean(hybrid_predictions > 0) * 100
    finite_pressure_rate = np.mean(np.isfinite(hybrid_predictions)) * 100
    correction_ratio = hybrid_predictions / np.maximum(physics_test_pressure, LOG_FLOOR)
    corr_q = np.percentile(correction_ratio, [1, 5, 50, 95, 99])
    print(f'     压力正值率: {positive_pressure_rate:.2f}% | 有限值率: {finite_pressure_rate:.2f}%')
    print(f'     修正倍率 P_hybrid/P_vdW-P: P01={corr_q[0]:.3f}, P05={corr_q[1]:.3f}, P50={corr_q[2]:.3f}, P95={corr_q[3]:.3f}, P99={corr_q[4]:.3f}')
    if len(delta_h_values) > 0:
        delta_h_arr = np.array(delta_h_values)
        delta_h_nonpositive_rate = delta_h_nonpositive_count / len(delta_h_arr) * 100
        print(f'     ΔH统计(kJ/mol): min={np.min(delta_h_arr):.2f}, P50={np.median(delta_h_arr):.2f}, max={np.max(delta_h_arr):.2f}, 非正值比例={delta_h_nonpositive_rate:.2f}%')
    else:
        print('     ΔH统计(kJ/mol): 无有效样本')
    if theta_eval_count > 0:
        theta_small_arr = np.array(theta_small_values)
        theta_large_arr = np.array(theta_large_values)
        theta_violation_rate = theta_bound_violations / (2 * theta_eval_count) * 100
        print(f'     θ小笼范围: [{np.min(theta_small_arr):.4f}, {np.max(theta_small_arr):.4f}] | θ大笼范围: [{np.min(theta_large_arr):.4f}, {np.max(theta_large_arr):.4f}]')
        print(f'     θ物理边界违反率(0≤θ≤1): {theta_violation_rate:.2f}% (n={theta_eval_count}×2个笼型)')
    else:
        print('     θ统计: 无有效样本')
    print('\n[10/10] Saving models...')
    try:
        joblib.dump(lgb_model, 'hybrid_lgbm_model.joblib')
        joblib.dump(rf_model, 'hybrid_rf_model.joblib')
        joblib.dump(scaler_X, 'hybrid_scaler_X.joblib')
        model_config = {'ensemble_weights': ensemble_weights, 'feature_names': feature_names, 'component_order': component_cols, 'log_floor': LOG_FLOOR, 'training_temp_range': (min_temp_train, max_temp_train), 'training_pressure_range': (dataset['p(MPa)'].min(), dataset['p(MPa)'].max()), 'model_version': '3.0', 'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'best_lgb_params': gscv.best_params_}
        joblib.dump(model_config, 'hybrid_model_config.joblib')
        export_supplementary_model_files(lgb_model, rf_model, scaler_X, model_config, params_file=params_file)
        print(' 模型文件保存成功')
    except Exception as e:
        print(f' 模型保存失败: {e}')
        return
    test_set_output = pd.DataFrame({'T(K)': valid_temperatures[test_indices], 'p_true(MPa)': true_test_pressure, 'p_physics(MPa)': physics_test_pressure, 'p_hybrid(MPa)': hybrid_predictions, 'structure': selected_structures[test_indices], 'log_residual': y_test, 'predicted_log_residual': predicted_log_residuals})
    test_compositions_df = pd.DataFrame(valid_compositions[test_indices], columns=component_cols, index=test_set_output.index)
    final_output_df = pd.concat([test_set_output, test_compositions_df], axis=1)
    output_csv_path = 'test_set_predictions_detailed_corrected.csv'
    final_output_df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
    print(f"Test predictions saved: {output_csv_path}")
    print('Training complete.')

def predict_pressure_for_new_conditions(temperature, composition_dict, model_path_prefix='hybrid', params_file='hydrate_parameters.xlsx', inhibitor_type=None, inhibitor_weight_fraction=0.0):
    try:
        lgbm_model = joblib.load(f'{model_path_prefix}_lgbm_model.joblib')
        rf_model = joblib.load(f'{model_path_prefix}_rf_model.joblib')
        scaler_X = joblib.load(f'{model_path_prefix}_scaler_X.joblib')
        config = joblib.load(f'{model_path_prefix}_model_config.joblib')
        ensemble_weights = config['ensemble_weights']
        feature_names = config['feature_names']
        component_order_map = {name: i for i, name in enumerate(config['component_order'])}
        LOG_FLOOR = config['log_floor']
        min_temp_train, max_temp_train = config.get('training_temp_range', (None, None))
    except Exception as e:
        print(f'Model load failed: {e}. Run train_and_save_models() first.')
        return None
    if min_temp_train and not (min_temp_train <= temperature <= max_temp_train):
        print(f'Warning: T={temperature} K outside training range [{min_temp_train:.2f}, {max_temp_train:.2f}] K')
    active_components = list(composition_dict.keys())
    active_composition = list(composition_dict.values())
    total_fraction = sum(active_composition)
    if abs(total_fraction - 1.0) > 1e-06:
        active_composition = [x / total_fraction for x in active_composition]
    try:
        physics_pressure, structure, _, _ = predict_structure_and_pressure(
            temperature, active_components, active_composition, params_file=params_file)
        if physics_pressure is None:
            print('Physics model failed.')
            return None
        structure_name = 'sI' if structure == 1 else 'sII'
    except Exception as e:
        print(f'Physics model error: {e}')
        return None
    composition_array = np.zeros(len(component_order_map))
    component_name_map = {
        'Methane': 'xCH4', 'Ethane': 'xC2H6', 'Propane': 'xC3H8', 'Carbon dioxide': 'xCO2',
        'Nitrogen': 'xN2', 'Hydrogen Sulfide': 'xH2S', 'i-Butane': 'xi-C4H10',
    }
    for name, frac in composition_dict.items():
        col_name = component_name_map.get(name)
        if col_name in component_order_map:
            composition_array[component_order_map[col_name]] = frac / total_fraction
    features_np = create_enhanced_features(
        np.array([temperature]), composition_array.reshape(1, -1),
        np.array([physics_pressure]), np.array([structure]))
    features_df = pd.DataFrame(features_np, columns=feature_names)
    features_scaled = pd.DataFrame(scaler_X.transform(features_df), columns=feature_names)
    try:
        lgb_pred = lgbm_model.predict(features_scaled)[0]
        rf_pred = rf_model.predict(features_scaled)[0]
        log_residual = ensemble_weights[0] * lgb_pred + ensemble_weights[1] * rf_pred
        final_pressure = exp(np.log(np.maximum(physics_pressure, LOG_FLOOR)) + log_residual)
        correction_pct = (final_pressure / physics_pressure - 1) * 100
    except Exception as e:
        print(f'Hybrid prediction failed: {e}')
        return None
    results = {
        'pressure_physics': physics_pressure,
        'pressure_hybrid': final_pressure,
        'structure': structure_name,
        'structure_id': structure,
        'correction_pct': correction_pct,
    }
    try:
        results['delta_H_kJ_mol'] = compute_formation_enthalpy(
            temperature, active_components, active_composition, structure, params_file)
    except Exception:
        results['delta_H_kJ_mol'] = None
    try:
        comp_params = [load_component_data(c, params_file) for c in active_components]
        theta, _ = compute_cage_occupancy(
            temperature, final_pressure, comp_params, active_composition, structure)
        results['theta_small_cage'] = float(np.sum(theta[0, :]))
        results['theta_large_cage'] = float(np.sum(theta[1, :]))
        results['theta_per_component'] = {
            active_components[k]: {'small': float(theta[0, k]), 'large': float(theta[1, k])}
            for k in range(len(active_components))
        }
    except Exception:
        results['theta_small_cage'] = None
        results['theta_large_cage'] = None
    results['water_activity'] = compute_water_activity(
        temperature, inhibitor_type, inhibitor_weight_fraction)
    print(f'T={temperature} K | P_vdW-P={physics_pressure:.4f} MPa | P_hybrid={final_pressure:.4f} MPa ({structure_name})')
    if results.get('delta_H_kJ_mol') is not None:
        print(f'dH={results["delta_H_kJ_mol"]:.2f} kJ/mol | theta_s={results.get("theta_small_cage")} theta_l={results.get("theta_large_cage")} | aw={results["water_activity"]:.4f}')
    return results


if __name__ == '__main__':
    RUN_MODE = 'predict'
    if RUN_MODE in ('train', 'both'):
        train_and_save_models()
    if RUN_MODE in ('predict', 'both'):
        predict_pressure_for_new_conditions(temperature=277.15, composition_dict={'Methane': 1.0})
