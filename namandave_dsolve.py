#!/usr/bin/env python
# coding: utf-8



from sympy import *
from math import *
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge




def eq_solve(differential_equation, function, independent_variable,              ics, display_=False):
    
    ans = dsolve(differential_equation, function,                  ics={function.subs(independent_variable, ics[0]):ics[1]})
    
    if display_:
        display(ans)
        
    return str(ans).split(',')[1].strip()[:-1]




def eq_solve_ideal(differential_equation, function, independent_variable,                    ics, range_, t_step, display_=False):
    
    ans = eq_solve(differential_equation, function, independent_variable,                    ics, display_)
    t = range_[0]
    true_val = []
    
    while t <= range_[1]:
        t += t_step
        true_val.append(round(eval(ans), 8))
        
    return np.array(true_val)




def _euler(differential_equation, dependent_variable, independent_variable,            ics, range_, step):
    
    f = lambdify(dependent_variable, differential_equation)
    d_f = lambdify(dependent_variable,                    differential_equation.diff(dependent_variable))
    d2_f = lambdify(dependent_variable,                     differential_equation.diff(dependent_variable, 2))
    
    start = range_[0]
    end = range_[1]
    x1 = ics
    x3 = ics
    x2 = ics
    t = start
    x1_arr = []
    x2_arr = []
    x3_arr = []
    t_arr = []
    
    while t <= end:
        x1 = x1 + f(x1)*step
        x2 = x2 + f(x2)*step + f(x2)*d_f(x2)*(step**2)/2
        x3 = x3 + f(x3)*step + f(x3)*d_f(x3)*(step**2)/2 +         ((f(x3)*(d_f(x3)**2)) + ((f(x3)**2)*d2_f(x3)))*(step**3)/6
        x1_arr.append(x1)
        x2_arr.append(x2)
        x3_arr.append(x3)
        t += step
        t_arr.append(t)
          
    return np.array(x1_arr), np.array(x2_arr), np.array(x3_arr), np.array(t_arr)




def _euler_na(differential_equation, dependent_variable,               independent_variable, ics, range_, step):
    
    
    f = lambdify((dependent_variable, independent_variable), differential_equation)
    g = diff(differential_equation, independent_variable) +     differential_equation*diff(differential_equation, dependent_variable)
    g_ = lambdify((dependent_variable, independent_variable), g)
    h = g.diff(independent_variable) + differential_equation*g.diff(dependent_variable)
    h_ = lambdify((dependent_variable, independent_variable), h)
    
    
    start = range_[0]
    end = range_[1]
    x1 = ics
    x3 = ics
    x2 = ics
    t = start
    x1_arr = []
    x2_arr = []
    x3_arr = []
    t_arr = []
    
    while t <= end:
        x1 = x1 + f(x1, t)*step
        x2 = x2 + f(x2, t)*step + g_(x2, t)*(step**2)/2
        x3 = x3 + f(x3, t)*step + g_(x3, t)*(step**2)/2 + h_(x3, t)*(step**3)/6
        x1_arr.append(x1)
        x2_arr.append(x2)
        x3_arr.append(x3)
        t += step
        t_arr.append(t)
        
    return np.array(x1_arr), np.array(x2_arr), np.array(x3_arr), np.array(t_arr)




def print_error(array, text):
    
    if text:
        print("\nError (", text, "):")
    else:
        print("Error analysis:")
        
    min_err = np.min(array)
    max_err = np.max(array)
    
    if min_err >= 0:
        pass
    elif max_err <= 0:
        min_err, max_err = max_err, min_err
    else:
        min_abs_err = np.min(np.abs(array))
        max_abs_err = np.max(np.abs(array))
        if min_abs_err in array:
            min_err = min_abs_err
        else:
            min_err = -min_abs_err
        
        if max_abs_err in array:
            max_err = max_abs_err
        else:
            max_err = -max_abs_err
            
    print("Local error: ", np.sum(np.abs(array)))
    print("Global error:", np.linalg.norm(array)/sqrt(len(array)))
    print("Mean of error: ", np.mean(array))
    print("Standard deviation: ", np.std(array))
    print("Minimum error: ", min_err)
    print("Maximum error: ", max_err, '\n')




def print_steps(ans, steps):
    
    print(ans[0][:steps], ans[1][:steps], ans[2][:steps],           ans[3][:steps], ans[4][:steps], ans[5][:steps],           ans[6][:steps], ans[7][:steps], sep='\n')
    




def test(function, dependent_variable, independent_variable,          ics, range_, step, analysis=True, show_steps=0,          diff_eq=None, function_=None, inde_var=None,          solve_symbolically=False,          true_function=None, true_array=None,          plot_val=False, display=False, save_fig=None,          base_solver=_euler,         xlabel=None, ylabel=None, error_ylabel=None,          **fig_kwargs):
    
    '''
    *args.validate(), **kwargs.validate() ommited. Be careful.
    '''
    a, b, c, t = base_solver(function, dependent_variable, independent_variable, ics, range_, step)
    
    ideal = true_array
    if true_function != None:
        ideal = [true_function(i) for i in t]
    elif solve_symbolically:
        ideal = eq_solve_ideal(diff_eq, function_, inde_var, ics=[range_[0], ics], display_=display, range_=range_, t_step=step)
        
    
    error1 = ideal - a
    error2 = ideal - b
    error3 = ideal - c
    
    if not ylabel:
        ylabel = r"$x(t)$"
    if not xlabel:
        xlabel = r"$t$"
    if not error_ylabel:
        error_ylabel = "Error"
    
    if plot_val == True:
        
        f, ax = plt.subplots(1, 2, **fig_kwargs)
        #f, ax = plt.subplots(1, 2, figsize=(30, 10))
        ax[0].plot(t, a, 'r')
        ax[0].plot(t, b, 'b')
        ax[0].plot(t, c, 'g')
        ax[0].plot(t, ideal, 'y')
        ax[0].legend(['Euler', 'Taylor (degree: 2)', 'Taylor (degree: 3)', 'Ideal'])
        ax[1].plot(t, error1, 'r')
        ax[1].plot(t, error2, 'b')
        ax[1].plot(t, error3, 'g')
        ax[1].legend([error_ylabel + ' Euler', error_ylabel + ' Taylor (degree: 2)', error_ylabel + ' Taylor (degree: 3)'])
        ax[0].set_xlabel(xlabel, fontsize=18)
        ax[0].set_ylabel(ylabel, fontsize=18)
        ax[1].set_xlabel(xlabel, fontsize=18)
        ax[1].set_ylabel(ylabel, fontsize=18)
        ax[0].grid()
        ax[1].grid()
        for ax1 in ax:
            for tick in ax1.xaxis.get_major_ticks():
                tick.label.set_fontsize(14) 
            for tick in ax1.yaxis.get_major_ticks():
                tick.label.set_fontsize(14) 
        plt.show()
        
        if save_fig:
            f.savefig(save_fig, dpi=300)
            
    elif plot_val  == 'error':
        
        f, ax = plt.subplots(1, 3, **fig_kwargs)
        #f, ax = plt.subplots(1, 3, figsize=(40, 10))
        ax[0].plot(t, error1, 'r')
        ax[1].plot(t, error2, 'b')
        ax[2].plot(t, error3, 'g')
        ax[0].set_xlabel(xlabel, fontsize=18)
        ax[0].set_ylabel(error_ylabel + " Euler", fontsize=18)
        ax[1].set_xlabel(xlabel, fontsize=18)
        ax[1].set_ylabel(error_ylabel + " Taylor (degree: 2)", fontsize=18)
        ax[2].set_xlabel(xlabel, fontsize=18)
        ax[2].set_ylabel(error_ylabel + " Taylor (degree: 3)", fontsize=18)
        ax[0].grid()
        ax[1].grid()
        ax[2].grid()
        for ax1 in ax:
            for tick in ax1.xaxis.get_major_ticks():
                tick.label.set_fontsize(14) 
            for tick in ax1.yaxis.get_major_ticks():
                tick.label.set_fontsize(14) 
        plt.show()
        if save_fig:
            f.savefig(save_fig, dpi=100)
            
    if analysis:
        print_error(error1, "Euler")
        print_error(error2, "Taylor (degree: 2)")
        print_error(error3, "Taylor (degree: 3)")
        
    if show_steps:
        print_steps(ans, show_steps)
        
        
    return t, a, b, c, ideal, error1, error2, error3




def give_points(arr, x, tol=1e-12):
    
    stable = []
    unstable = []
    ND = []
    tot_points_id = []
    
    for i in range(len(arr)):
        if abs(arr[i]) < tol:
            if arr[i - 1] > 0 and arr[i + 1] < 0:
                stable.append(round(x[i], 3))
            elif arr[i - 1] < 0 and arr[i + 1] > 0:
                unstable.append(round(x[i], 3))
            else:
                ND.append(round(x[i], 3))
            tot_points_id.append(i)
            
    return list(set(map(lambda x: round(x, 3), stable))), list(set(map(lambda x: round(x, 3), unstable))),         list(set(map(lambda x: round(x, 3), ND))), tot_points_id




def _euler_over_flow_controlled(differential_equation, dependent_variable,                                 independent_variable, ics, range_, step, penalty=5):
    
    f = lambdify(dependent_variable, differential_equation)
    d_f = lambdify(dependent_variable, differential_equation.diff(dependent_variable))
    d2_f = lambdify(dependent_variable, differential_equation.diff(dependent_variable, 2))
    
    start = range_[0]
    end = range_[1]
    x1 = ics
    x3 = ics
    x2 = ics
    t = start
    x1_arr = []
    x2_arr = []
    x3_arr = []
    t_arr = []
    
    while t <= end:
        try:
            x1 = x1 + f(x1)*step
            x2 = x2 + f(x2)*step + f(x2)*d_f(x2)*(step**2)/2
            x3 = x3 + f(x3)*step + f(x3)*d_f(x3)*(step**2)/2 + ((f(x3)*(d_f(x3)**2)) + ((f(x3)**2)*d2_f(x3)))*(step**3)/6
        except:
            print("Overflow detected at ics:", ics, "\npenalty:",penalty,                   " rearmost value of array:", x3_arr[-penalty - 1])
            if penalty != 0:
                return np.array(x1_arr[:-penalty]), np.array(x2_arr[:-penalty]), np.array(x3_arr[:-penalty]), np.array(t_arr[:-penalty])
            else:
                break
        x1_arr.append(x1)
        x2_arr.append(x2)
        x3_arr.append(x3)
        t += step
        t_arr.append(t)
    return np.array(x1_arr), np.array(x2_arr), np.array(x3_arr), np.array(t_arr)




def test_phase(function, dependent_variable, independent_variable, ics_arr, range_,                step, phase_range, phase_step, plot_val="val", save_fig=None,                tol=1e-12, dpi=100, dpi_phase=100, calculate_points=False,                stable=None, unstable=None, ND=None, tick_points=None,                markersize=160, bbox_to_anchor=None, ND_arr=None, penalty=5,                leg_stable=None, leg_unstable=None, leg_ND=None,                ics_arr_sym=None, legend_fontsize=None,                xlabel=None, ylabel=None, **fig_kwargs):
    
    eu_arr = []
    ty2_arr = []
    ty3_arr = []
    t_arr = []
    d_f = lambdify(dependent_variable, function)
    
    for ics in ics_arr:
        
        a, b, c, t = _euler_over_flow_controlled(function, dependent_variable,                                                  independent_variable, ics,                                                  range_, step, penalty)
        eu_arr.append(a)
        ty2_arr.append(b)
        ty3_arr.append(c)
        t_arr.append(t)

        
    if plot_val=="phase":
        x = list(np.arange(-1.0001, -0.999, 0.0001))
        ideal_d_f = [d_f(i) for i in x]
        ideal_d_f_eu = [d_f(i) for i in a]
        ideal_d_f_ty1 = [d_f(i) for i in b]
        ideal_d_f_ty2 = [d_f(i) for i in c]
        f, ax = plt.subplots(1, 1, **fig_kwargs)
        ax.plot(x, ideal_d_f)
        ax.scatter(a, ideal_d_f_eu)
        ax.scatter(b, ideal_d_f_ty1)
        ax.scatter(c, ideal_d_f_ty2)
        ax.legend(['Ideal', 'Euler', 'Taylor (degree: 2)', 'Taylor (degree: 3)'])
        ax.set_ylabel("dx/dt", fontsize=18)
        ax.set_xlabel("x", fontsize=18)
        plt.grid()
        print(ideal_d_f_eu[-1], ideal_d_f_ty1[-1], ideal_d_f_ty2[-1])
        
    elif plot_val == "val":
        f, ax = plt.subplots(1, 1, **fig_kwargs)
        x = np.arange(phase_range[0], phase_range[1], phase_step)
        d_x = [d_f(i) for i in x]
        if calculate_points:
            stable, unstable, ND, tot_points_id = give_points(d_x, x, tol=tol)
        if not leg_stable and stable != []:
            print("leg_stable depricated")
            leg_stable = ",".join(map(lambda x: str(round(x, 2)), stable))
        if not leg_unstable and unstable != []:
            print("leg_unstable depricated")
            leg_unstable = ",".join(map(lambda x: str(round(x, 2)), unstable))
        if not leg_ND and ND != []:
            print("leg_ND depricated")
            leg_ND = ",".join(map(lambda x: str(round(x, 2)), ND))
        print(stable, unstable, ND)
        if stable != []:
            ax.scatter(stable, np.zeros(len(stable)), s = markersize,                        c="black", label=leg_stable)
        if unstable != []:
            ax.scatter(unstable, np.zeros(len(unstable)), s = markersize,                        marker="o", facecolors='none', edgecolors='black',                        label=leg_unstable)
        if ND != []:
            marker_style = dict(color='black', linestyle=':', marker='o',                                 markersize=markersize//12,                                 markerfacecoloralt='white')
            if not ND_arr:
                ax.plot(ND, np.zeros(len(ND)), fillstyle="right",                         **marker_style, label=leg_ND)
            else:
                for i in range(len(ND)):
                    ax.plot([ND[i]], [0], label=leg_ND,                             fillstyle=ND_arr[i], **marker_style)
        ax.legend(prop={'size': legend_fontsize})
        if calculate_points and not tick_points:
            tot_points_id.insert(0, 0)
            tot_points_id.append(len(x) - 1)
            for idx in range(len(tot_points_id) - 1):
                idx = (tot_points_id[idx] + tot_points_id[idx + 1])// 2
                if d_x[idx] > 0:
                    ax.scatter([x[idx]], [0], marker="$>$", c='k', s=markersize)
                elif d_x[idx] < 0:
                    ax.scatter([x[idx]], [0], marker="$<$", c='k', s=markersize)
        else:
            
            for point in tick_points:
                if d_f(point) > 0:
                    ax.scatter([point], [0], marker="$>$", c='k', s=markersize)
                else:
                    ax.scatter([point], [0], marker="$<$", c='k', s=markersize)
        ax.plot(x, d_x)
        plt.grid()
        plt.axhline(y=0, color='k')
        plt.axvline(x=0, color='k')
        if xlabel:
            ax.set_xlabel(xlabel, fontsize=18)
        else:
            ax.set_xlabel(r'$x$', fontsize=18)
        if ylabel:
            ax.set_ylabel(ylabel, rotation=0, fontsize=18)
        else:
            ax.set_ylabel(r'$\dot{x}$', rotation=0, fontsize=18)
    
        ax.yaxis.set_label_coords(-0.06, 0.5)
        plt.title("Phase plot")
        if save_fig:
            plt.savefig(save_fig + "_phase.png", dpi=dpi_phase)
            
        if not ics_arr_sym:
            ics_arr_sym = list(map(lambda x: round(x, 2), ics_arr))
        plt.figure(**fig_kwargs)
        for i in range(len(eu_arr)):
            plt.plot(t_arr[i], eu_arr[i])
        plt.grid()
        plt.legend(ics_arr_sym, bbox_to_anchor=bbox_to_anchor)
        plt.title("Euler analysis plot")
        plt.ylabel(r'$x$', rotation=0, fontsize=18)
        plt.xlabel(r'$t$', fontsize=18)
        if save_fig:
            plt.savefig(save_fig + "_eu.png", dpi=dpi)
        
        plt.figure(**fig_kwargs)
        for i in range(len(ty2_arr)):
            plt.plot(t_arr[i], ty2_arr[i])
        plt.grid()
        plt.legend(ics_arr_sym, bbox_to_anchor=bbox_to_anchor)
        plt.title("Taylor (D:2) analysis plot")
        plt.ylabel(r'$x$', rotation=0, fontsize=18)
        plt.xlabel(r'$t$', fontsize=18)
        if save_fig:
            plt.savefig(save_fig + "_ty2.png", dpi=dpi)
        
        
        plt.figure(**fig_kwargs)
        for i in range(len(ty3_arr)):
            plt.plot(t_arr[i], ty3_arr[i])
        plt.grid()
        plt.legend(ics_arr_sym, bbox_to_anchor=bbox_to_anchor)
        plt.title("Taylor (D:3) analysis plot")
        plt.ylabel(r'$x$', rotation=0, fontsize=18)
        plt.xlabel(r'$t$', fontsize=18)
        if save_fig:
            plt.savefig(save_fig + "_ty3.png", dpi=dpi)
        
        




t = Symbol('t')
x = Function('x')(t)
f = Function('f')(x)
f = x**2 - 1
ics_arr = [1.000001, 0.9999, 0, -0.9999, -1.001, -3]
ics_arr.sort(reverse=True)
ics_arr_sym = ics_arr
range_ = [0, 8]
time_step = 0.1
range_phase = [-2, 2]
phase_step = 0.001
test_phase(f, x, t, ics_arr, range_, time_step, range_phase,            phase_step, figsize=(10, 10),save_fig="S41A",            calculate_points=True, ics_arr_sym=ics_arr_sym)




t = Symbol('t')
x = Function('x')(t)
f = Function('f')(x)
f = -x**3
ics_arr = [0.001, -0.015, 1, -1]
ics_arr.sort(reverse=True)
ics_arr_sym = ics_arr
range_ = [0, 2000]
time_step = 0.1
range_phase = [-5, 5]
phase_step = 0.001
l_temp = [-3, 3]
test_phase(f, x, t, ics_arr, range_, time_step, range_phase, phase_step,            figsize=(12, 12),            save_fig="S41B",            stable=[0], unstable=[], ND=[], tick_points=l_temp, ics_arr_sym=ics_arr_sym)




t = Symbol('t')
x = Function('x')(t)
f = Function('f')(x)
f = x**3
ics_arr = [0.105, -0.105, 0.11, -0.11]
ics_arr.sort(reverse=True)
ics_arr_sym = ics_arr
range_ = [0, 40]
time_step = 0.01
range_phase = [-5, 5]
phase_step = 0.001
stable = []
unstable = [0]
ND = []
#ND_arr = ['left']
ticks = [-3, 3]
test_phase(f, x, t, ics_arr, range_, time_step, range_phase, phase_step,            figsize=(12, 12),            save_fig="S41C",            stable=stable, unstable=unstable, ND=ND, tick_points=ticks, ics_arr_sym=ics_arr_sym)#, ND_arr=ND_arr)




t = Symbol('t')
x = Function('x')(t)
f = Function('f')(x)
f = x**2
ics_arr = [-0.2, -0.1, -0.017, -0.012, 0.008, 0.00805]
ics_arr.sort(reverse=True)
ics_arr_sym = ics_arr
range_ = [0, 123.5]
time_step = 0.001
range_phase = [-2, 2]
phase_step = 0.001
stable = []
unstable = []
ND = [0]
ND_arr = ['left']
ticks = [-1, 1]
test_phase(f, x, t, ics_arr, range_, time_step, range_phase, phase_step,            figsize=(10, 10),            save_fig="S41D",            stable=stable, unstable=unstable, ND=ND, tick_points=ticks, ND_arr=ND_arr, ics_arr_sym=ics_arr_sym)




t = Symbol('t')
x = Function('x')(t)
f = Function('f')(x)
f = 4*x**2 - 16
ics_arr = [-3, -2.001, 0, -1, 1, 2.001, 2.0011, 1.9]
ics_arr.sort(reverse=True)
ics_arr_sym = ics_arr
range_ = [0, 0.5]
time_step = 0.001
range_phase = [-4, 4]
phase_step = 0.001
stable = [-2]
unstable = [2]
ND = []
ND_arr = []
ticks = [0, 3, -3]
test_phase(f, x, t, ics_arr, range_, time_step, range_phase, phase_step,            figsize=(10, 10),            save_fig="S41E",            stable=stable, unstable=unstable, ND=ND, tick_points=ticks, ND_arr=ND_arr, ics_arr_sym=ics_arr_sym)




t = Symbol('t')
x = Function('x')(t)
f = Function('f')(x)
f = 1 - x**14
ics_arr = [-0.999, 0, 1.2, 0.8, -1.00001, -1.000001]
ics_arr.sort(reverse=True)
ics_arr_sym = ics_arr
range_ = [0, 2.5]
time_step = 0.0001
range_phase = [-1.1, 1.1]
phase_step = 0.001
stable = [1]
unstable = [-1]
ND = []
ND_arr = []
ticks = [0, -1.2, 1.2]
test_phase(f, x, t, ics_arr, range_, time_step, range_phase, phase_step,            figsize=(10, 10),            save_fig="S41F",            stable=stable, unstable=unstable, ND=ND, tick_points=ticks,            ND_arr=ND_arr, penalty=2, ics_arr_sym=ics_arr_sym)




t = Symbol('t')
x = Function('x')(t)
f = Function('f')(x)
f = x - x**3
ics_arr = [-4, 4, -1.5, 0.5, 0.01, -0.01, 0.1, -0.1, 0.5, 1.5]
ics_arr.sort(reverse=True)
range_ = [0, 7]
time_step = 0.01
range_phase = [-1.8, 1.8]
phase_step = 0.001
stable = [1, -1]
unstable = [0]
ND = []
ND_arr = []
ticks = [0.5, -0.5, 1.5, -1.5]
test_phase(f, x, t, ics_arr, range_, time_step, range_phase,            phase_step, figsize=(10, 10),            save_fig="S41G",            stable=stable, unstable=unstable, ND=ND,            tick_points=ticks, ND_arr=ND_arr, penalty=0)




t = Symbol('t')
x = Function('x')(t)
f = Function('f')(x)
f = sp.exp(-x)*sp.sin(x)
ics_arr = [-0.001, 0.2, 0.9, 2, pi + 1, pi + 1.5, 2*pi + 0.001][::-1]
ics_arr_sym = ["-0.001", "0.2", "0.9", "2", "$\pi + 1$",                "$\pi + 1.5$", "$2\pi + 0.001$"][::-1]
range_ = [0, 150]
time_step = 0.05
range_phase = [-0.2, 7]
phase_step = 0.001
leg_stable = r"$\pi\hspace{2}(2k+1)\pi$"
leg_unstable = r"$0,2\pi\hspace{2}2k\pi$"
test_phase(f, x, t, ics_arr, range_, time_step, range_phase,            phase_step, figsize=(10, 10), save_fig="S41H",            stable=[pi], unstable=[0, 2*pi], ND=[],            tick_points=[-0.4, pi/2, 3*pi/2, 2*pi + 0.4],            bbox_to_anchor=(0.9, 0.9),            leg_stable=leg_stable, leg_unstable=leg_unstable,            ics_arr_sym=ics_arr_sym, legend_fontsize=15)




t = Symbol('t')
x = Function('x')(t)
f = Function('f')(x)
f = 1 + 0.5*sp.sin(x)
ics_arr = list(range(-4, 5))

range_ = [0, 20]
time_step = 0.01
range_phase = [-5, 5]
phase_step = 0.001
test_phase(f, x, t, ics_arr, range_, time_step,            range_phase, phase_step, figsize=(12, 12),            save_fig="S41I",            stable=[], unstable=[], ND=[],            tick_points=[-2.5, 2.5], )




t = Symbol('t')
x = Function('x')(t)
f = Function('f')(x)
f = 1 - 2*sp.cos(x)
temp = pi/3
temp2 = 2*pi
ics_arr_tmp = [temp + 0.01, temp - 0.01, temp + temp2 + 0.01,            temp + temp2 - 0.01, temp - temp2 + 0.01,            temp - temp2 - 0.01]
ics_arr_sym_tmp = [r"$ \frac{\pi}{3} + 0.01$", r"$ \frac{\pi}{3} - 0.01$",                r"$ \frac{7\pi}{3} + 0.01$", r"$ \frac{7\pi}{3} - 0.01$",                r"$ \frac{-5\pi}{3} + 0.01$", r"$ \frac{-5\pi}{3} - 0.01$", ]
ics_arr = sorted(list(ics_arr_tmp), reverse=True)
ics_arr_sym = [ics_arr_sym_tmp[ics_arr_tmp.index(i)] for i in ics_arr]
range_ = [0, 7]
time_step = 0.01
range_phase = [-10, 10]
phase_step = 0.001
stable = [-pi/3, -pi/3 - 2*pi, -pi/3 + 2*pi]
leg_stable = r"$\frac{-7\pi}{3},\frac{-\pi}{3},\frac{5\pi}{3}\Rightarrow\frac{(6k - 1)}{3}\pi$"
unstable = [pi/3, pi/3 + 2*pi, pi/3 - 2*pi]
leg_unstable = r"$\frac{-5\pi}{3},\frac{\pi}{3},\frac{7\pi}{3}\Rightarrow\frac{(6k + 1)}{3}\pi$"
ND = []
ND_arr = []
ticks = [0, pi, -pi, -2*pi, 2*pi, 3*pi, -3*pi]
test_phase(f, x, t, ics_arr, range_, time_step,            range_phase, phase_step, figsize=(10, 10),            save_fig="S41Ja",            stable=stable, unstable=unstable, ND=ND,            tick_points=ticks, ND_arr=ND_arr, penalty=0,            ics_arr_sym=ics_arr_sym, leg_stable=leg_stable,            leg_unstable=leg_unstable, legend_fontsize=14)




t = Symbol('t')
x = Function('x')(t)
f = Function('f')(x)
f = 1 - 2*sp.cos(x)
temp = pi/3
temp2 = 2*pi
ics_arr_tmp = [temp + 0.01, temp - 0.01, temp + temp2 + 0.01,            temp + temp2 - 0.01, temp - temp2 + 0.01,            temp - temp2 - 0.01, (temp + temp2/2)/2 + 0.01,            2*pi - 0.01, 0.01, -pi - 0.01, 3*pi, -2*pi]
ics_arr_sym_tmp = [r"$ \frac{\pi}{3} + 0.01$", r"$ \frac{\pi}{3} - 0.01$",                r"$ \frac{7\pi}{3} + 0.01$", r"$ \frac{7\pi}{3} - 0.01$",                r"$ \frac{-5\pi}{3} + 0.01$", r"$ \frac{-5\pi}{3} - 0.01$",                r"$ \frac{2\pi}{3} + 0.01$", r"$ 2\pi - 0.01$",                r"$ 0.01$", r"$ -\pi - 0.01$", r"$3\pi$", r"$-2\pi$" ]
ics_arr = sorted(list(ics_arr_tmp), reverse=True)
ics_arr_sym = [ics_arr_sym_tmp[ics_arr_tmp.index(i)] for i in ics_arr]
range_ = [0, 7]
time_step = 0.01
range_phase = [-10, 10]
phase_step = 0.001
stable = [-pi/3, -pi/3 - 2*pi, -pi/3 + 2*pi]
leg_stable = r"$\frac{-7\pi}{3},\frac{-\pi}{3},\frac{5\pi}{3}\Rightarrow\frac{(6k - 1)}{3}\pi$"
unstable = [pi/3, pi/3 + 2*pi, pi/3 - 2*pi]
leg_unstable = r"$\frac{-5\pi}{3},\frac{\pi}{3},\frac{7\pi}{3}\Rightarrow\frac{(6k + 1)}{3}\pi$"
ND = []
ND_arr = []
ticks = [0, pi, -pi, -2*pi, 2*pi, 3*pi, -3*pi]
test_phase(f, x, t, ics_arr, range_, time_step,            range_phase, phase_step, figsize=(10, 10),            save_fig="S41Jb",            stable=stable, unstable=unstable, ND=ND,            tick_points=ticks, ND_arr=ND_arr, penalty=0,            ics_arr_sym=ics_arr_sym, leg_stable=leg_stable,            leg_unstable=leg_unstable, legend_fontsize=15,            bbox_to_anchor=(1, 1))




t = Symbol('t')
x = Function('x')(t)
f = Function('f')(x)
f = sp.exp(x) - sp.cos(x)
ics_arr = [-5, -4.8, -4.2, -2.5, -0.6,            -0.2, 0.0006, 0.001]
ics_arr.sort(reverse=True)
ics_arr_sym = ics_arr
range_ = [0, 6.55]
time_step = 0.01
range_phase = [-5, 1]
phase_step = 0.00001
tick_points = [1, -0.6, -3, -5]
test_phase(f, x, t, ics_arr, range_, time_step,            range_phase, phase_step, figsize=(12, 12),            save_fig="S41K",            tick_points=tick_points, calculate_points=True,            tol=5e-6, penalty=10, ics_arr_sym=ics_arr_sym)




t = Symbol('t')
x = Function('x')(t)
f = Function('f')(x)
f = x*(1 - x)
ics_arr = [-0.004, 0.1, 0.5, 2, 3]
ics_arr.sort(reverse=True)
ics_arr_sym = ics_arr
range_ = [0, 6.5]
time_step = 0.1
range_phase = [-1, 2]
phase_step = 0.01
stable = [1]
unstable = [0]
ND = []
ND_arr = []
ticks = [-0.5, 0.5, 1.5]
test_phase(f, x, t, ics_arr, range_, time_step,            range_phase, phase_step, figsize=(10, 10),            save_fig="S42A",            stable=stable, unstable=unstable, ND=ND,            tick_points=ticks, ND_arr=ND_arr, ics_arr_sym=ics_arr_sym)




t = Symbol('t')
x = Function('x')(t)
f = Function('f')(x)
f = x*(1 - x)*(2 - x)
ics_arr = [-0.001, 0.1, 0.5, 1.5, 1.99, 2.001]
ics_arr.sort(reverse=True)
ics_arr_sym = ics_arr
range_ = [0, 6.5]
time_step = 0.1
range_phase = [-0.6, 2.6]
phase_step = 0.01
stable = [1]
unstable = [0, 2]
ND = []
ND_arr = []
ticks = [-0.3, 0.5, 1.5, 2.3]
test_phase(f, x, t, ics_arr, range_, time_step,            range_phase, phase_step, figsize=(10, 10),            save_fig="S42B",            stable=stable, unstable=unstable, ND=ND,            tick_points=ticks, ND_arr=ND_arr, penalty=3, ics_arr_sym=ics_arr_sym)




t = Symbol('t')
x = Function('x')(t)
f = Function('f')(x)
f = sp.tan(x)
ics_arr = [-0.001, -0.0005, -0.00025, 0.00025, 0.0005, 0.001]
ics_arr.sort(reverse=True)
ics_arr_sym = ics_arr
range_ = [0.001, -log(sin(0.001)) - 0.005]
time_step = 0.001
range_phase = [-pi/2 + 0.2, pi/2 - 0.2]
phase_step = 0.01
stable = []
unstable = [0]
ND = []
ND_arr = []
ticks = [-pi/4 + 0.1, pi/4 - 0.1]
test_phase(f, x, t, ics_arr, range_, time_step,            range_phase, phase_step, figsize=(10, 10),            save_fig="S42C",            stable=stable, unstable=unstable, ND=ND,            tick_points=ticks, ND_arr=ND_arr, penalty=3,            ics_arr_sym=ics_arr_sym)




t = Symbol('t')
x = Function('x')(t)
f = Function('f')(x)
f = (x**2)*(6 - x)
ics_arr = [-2, -1, -0.2, 0.1, 0.15, 0.2, 0.5, 1, 4, 6.5, 7, 8, 10][::-1]
range_ = [0, 2]
time_step = 0.001
range_phase = [-2, 7]
phase_step = 0.01
stable = [6]
unstable = []
ND = [0]
ND_arr = ["left"]
ticks = [-1, 3, 6.5]
test_phase(f, x, t, ics_arr, range_, time_step,            range_phase, phase_step, figsize=(10, 10),            save_fig="S42D",            stable=stable, unstable=unstable, ND=ND,            tick_points=ticks, ND_arr=ND_arr, penalty=3,            bbox_to_anchor=(1, 1))




t = Symbol('t')
x = Function('x')(t)
f = Function('f')(x)
f = 1 - sp.exp(-x**2)
ics_arr = [-5, -4, -3, -2, -1, -0.5, -0.1, .1, .2, 0.5, 1, 2, 3][::-1]
range_ = [0, 20]
time_step = 0.01
range_phase = [-2, 2]
phase_step = 0.01
stable = []
unstable = []
ND = [0]
ND_arr = ["left"]
ticks = [-1, 1]
test_phase(f, x, t, ics_arr, range_, time_step,            range_phase, phase_step, figsize=(10, 10),            save_fig="S42E",            stable=stable, unstable=unstable, ND=ND,            tick_points=ticks, ND_arr=ND_arr, penalty=3,            bbox_to_anchor=(1, 1))




t = Symbol('t')
x = Function('x')(t)
f = Function('f')(x)
f = 1 - sp.exp(-x**2)
ics_arr = [-5, -4, -3, -2, -1, -0.5, -0.1, .1, .2, 0.5, 1, 2, 3][::-1]
range_ = [0, 20]
time_step = 0.01
range_phase = [-2, 2]
phase_step = 0.01
stable = []
unstable = []
ND = [0]
ND_arr = ["left"]
ticks = [-1, 1]
test_phase(f, x, t, ics_arr, range_, time_step,            range_phase, phase_step, figsize=(10, 10),            save_fig="S42E",            stable=stable, unstable=unstable, ND=ND,            tick_points=ticks, ND_arr=ND_arr, penalty=3,            bbox_to_anchor=(1, 1))




t = Symbol('t')
x = Function('x')(t)
f = Function('f')(x)
f = sp.log(x)
ics_arr = [0.80, 0.85, 0.9, 0.95, 1.05, 1.1, 2][::-1]
range_ = [0, 15]
time_step = 0.01
range_phase = [0.1, 5]
phase_step = 0.01
stable = []
unstable = [1]
ND = []
ND_arr = []
ticks = [0.5, 2.5]
test_phase(f, x, t, ics_arr, range_, time_step,            range_phase, phase_step, figsize=(10, 10),            save_fig="S42Fa",            stable=stable, unstable=unstable, ND=ND,            tick_points=ticks, ND_arr=ND_arr, penalty=3,            bbox_to_anchor=(1, 1))




t = Symbol('t')
x = Function('x')(t)
f = Function('f')(x)
f = sp.log(x)
ics_arr = [0.95, 0.96, 0.97, 1.05, 1.1, 2][::-1]
range_ = [0, 2.4]
time_step = 0.01
range_phase = [0.1, 5]
phase_step = 0.01
stable = []
unstable = [1]
ND = []
ND_arr = []
ticks = [0.5, 2.5]
test_phase(f, x, t, ics_arr, range_, time_step,            range_phase, phase_step, figsize=(10, 10),            save_fig="S52Fb",            stable=stable, unstable=unstable, ND=ND,            tick_points=ticks, ND_arr=ND_arr, penalty=3,            bbox_to_anchor=(1, 1))




t = Symbol('t')
x = Function('x')(t)
f = Function('f')(x)
a = -1
f = a*x - x**3
ics_arr = [-1, -0.5, -0.25, 0.25, 0.5, 1][::-1]
range_ = [0, 7]
time_step = 0.01
range_phase = [-3, 3]
phase_step = 0.01
stable = [0]
unstable = []
ND = []
ND_arr = []
ticks = [-1.5, 1.5]
test_phase(f, x, t, ics_arr, range_, time_step,            range_phase, phase_step, figsize=(10, 10),            save_fig="S42Ga",            stable=stable, unstable=unstable, ND=ND,            tick_points=ticks, ND_arr=ND_arr, penalty=3,            bbox_to_anchor=(1, 1))




t = Symbol('t')
x = Function('x')(t)
f = Function('f')(x)
a = 1
f = a*x - x**3
ics_arr = [-2, -1.5, -0.5, -0.1, 0.1, 0.5, 1.5, 2][::-1]
range_ = [0, 5]
time_step = 0.01
range_phase = [-2, 2]
phase_step = 0.01
stable = [-1, 1]
unstable = [0]
ND = []
ND_arr = []
ticks = [-1.5, -0.5, 0.5, 1.5]
test_phase(f, x, t, ics_arr, range_, time_step,            range_phase, phase_step, figsize=(10, 10),            save_fig="S42Gb",            stable=stable, unstable=unstable, ND=ND,            tick_points=ticks, ND_arr=ND_arr, penalty=3,            bbox_to_anchor=(1, 1))




t = Symbol('t')
x = Function('x')(t)
f = Function('f')(x)
m = 110 #kg
g = 9.8 #ms^-2
k = 0.18
A = g
B = k/m
f = A - B*x*x
t_f = lambda t: sqrt(A/B)*tanh(sqrt(A*B)*t)
ans = test(f, x, t, 0, [0, 50], 0.001, true_function = t_f,            plot_val=True, display=True, analysis=False,            save_fig="S51.png",            figsize=(24, 10), ylabel="$v(t)$")
ans = test(f, x, t, 0, [0, 50], 0.001, true_function = t_f,            plot_val='error', display=True, analysis=True,            save_fig="S51e.png", figsize=(40, 10))




t = Symbol('t')
x = Function('x')(t)
f = Function('f')(x)
v0 = 4
P = 400 #ms^-2
m = 70
b = 0.5
A = 0.33
rho = 1.25
f = P/(m*x)
t_f = lambda t: sqrt(v0**2 + (2*P*t/m))
ans = test(f, x, t, 4, [0, 100], 0.001, true_function = t_f,            plot_val=True, display=True, analysis=False,            save_fig="S52a.png", figsize=(24, 10),            ylabel="$v(t)$")
ans = test(f, x, t, 4, [0, 100], 0.001, true_function = t_f,            plot_val='error', display=True, analysis=True,            save_fig="S52ae.png", figsize=(40, 10))




# set 5 problem 2b
t = Symbol('t')
x = Function('x')(t)
f = Function('f')(x)
v0 = 4
P = 400 #ms^-2
m = 70
b = 0.5
A = 0.33
rho = 1.25
f = P/(m*x) - ((b*rho*A)/m)*x**2

a, _b, c, t = _euler(f, x, t, 4, [0, 100], 0.1)
plt.figure(figsize=(10, 10))
plt.plot(t, a)
plt.plot(t, _b)
plt.plot(t, c)
plt.grid()
plt.ylabel("$v(t)$", fontsize=18)
plt.xlabel("$t$", fontsize=18)
plt.legend(['Euler', 'Taylor (degree: 2)', 'Taylor (degree: 3)'])
plt.savefig("S52b.png", dpi=100)
print("Taylor (D:3) array's rearmost value:", c[-1])
print("Terminal velocity: ", (P/(b*rho*A))**(1/3))




t = Symbol('t')
x = Function('x')(t)
f = Function('f')(x)
v0 = 4
P = 400 #ms^-2
m = 70
b = 0.5
A = 0.33
rho = 1.25
f = P/(m*x) - ((b*rho*A)/m)*x**2

ics_arr = [4]
ics_arr.sort(reverse=True)
ics_arr_sym = ics_arr
range_ = [0, 60]
time_step = 0.1
range_phase = [0.4, 30]
phase_step = 0.01
stable = [(P/(b*rho*A))**(1/3)]
unstable = []
ND = []
ND_arr = []
leg_stable = r"$(\frac{P}{b\rho A})^\frac{1}{3}$"+"= 12.47"
ticks = [6, 23]
ylabel = r"$\dot{v}$"
xlabel = r"$v$"
test_phase(f, x, t, ics_arr, range_, time_step,            range_phase, phase_step, figsize=(10, 10),            save_fig="S42A",            stable=stable, unstable=unstable, ND=ND,            tick_points=ticks, ND_arr=ND_arr,            ics_arr_sym=ics_arr_sym,            leg_stable=leg_stable, legend_fontsize=15,            xlabel=xlabel, ylabel=ylabel)




# set 5 problem 2b
t = Symbol('t')
x = Function('x')(t)
f = Function('f')(x)
v0 = 4
P = 400 #ms^-2
m = 70
b = 0.5
A = 0.33
rho = 1.25
a = Symbol('a')
b = Symbol('b')
dsolve((x/(1 - x**3))*x.diff(t) - 1, x, ics={x.subs(t, 0):4})




t = Symbol('t')
x = Symbol('x')
f = Function('f')(x, t)
f = t**2 - x
y = Function('y')(t)
#dsolve(y.diff(t) + y - t**2, y, ics={y.subs(t, 0):1})
diff_eq = y.diff(t) + y - t**2
ans = test(f, x, t, 1, [0, 2], 0.01, display=True, base_solver=_euler_na,            plot_val=True, #save_fig="Output.png",\
           figsize=(20, 10), solve_symbolically=True, \
           function_=y, inde_var=t, diff_eq=diff_eq, analysis=False)
ans = test(f, x, t, 1, [0, 2], 0.01, display=True, base_solver=_euler_na,            plot_val=True, #save_fig="Output.png",\
           figsize=(20, 10), solve_symbolically=True, function_=y,\
           inde_var=t, diff_eq=diff_eq)




x = Symbol('x')
f = Function('f')(x)
f = -1*x*x
a, b, c, t_ = _euler(f, x, None, 1, [1, 2], 0.01)
ideal = 1/t_
ans = test(f, x, None, 1, [1, 2], 0.01, true_array=ideal,
           plot_val=True, display=True, #save_fig="Output.png",
            figsize=(20, 10))




t = Symbol('t')
x = Function('x')(t)
f = Function('f')(x)
f = -1*x*x
diff_eq = x.diff(t) + x*x  # = 0 explicitly
ans = test(f, x, None, 1, [1, 2], 0.01, solve_symbolically=True, function_=x,
            inde_var=t, diff_eq=x.diff(t) + x*x, plot_val=True, \
           display=True, #save_fig="Output.png",
            figsize=(20, 10))




t = Symbol('t')
x = Function('x')(t)
f = Function('f')(x)
f = -1*x*x
true_function = lambda t: 1/t
ans = test(f, x, None, 1, [1, 2], 0.01, true_function=true_function, plot_val=True, display=True, #save_fig="Output.png",
            figsize=(20, 10))




x = Symbol('x')
t = Symbol('t')
f = Function('f')(x, t)
f = -x + sp.exp(-t)
a, b, c, t_ = _euler_na(f, x, t, 1, [0, 2], 0.01)
ideal = (1 + t_)*np.exp(-t_)
ans = test(f, x, t, 1, [0, 2], 0.01, display=True, true_array=ideal, base_solver=_euler_na, plot_val=True, #save_fig="Output.png",
            figsize=(20, 10))




t = Symbol('t')
x = Symbol('x')
f = Function('f')(x, t)
f = -x + sp.exp(-t)
y = Function('y')(t) # depricated x as y

ans = test(f, x, t, 1, [0, 2], 0.01, display=True,            base_solver=_euler_na, plot_val=True, #save_fig="Output.png",
            figsize=(20, 10), solve_symbolically=True, function_=y,
            inde_var=t, diff_eq=y.diff(t) + y - sp.exp(-t))




t = Symbol('t')
x = Symbol('x')
f = Function('f')(x, t)
f = -x + sp.exp(-t)

ans = test(f, x, t, 1, [0, 2], 0.01, display=True,            true_function=lambda t: (t+1)*exp(-t),            base_solver=_euler_na, plot_val=True, #save_fig="Output.png",
            figsize=(20, 10))

