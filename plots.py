# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 18:27:55 2020

@author: nagsa
"""

import os 
import numpy as np
import matplotlib.pyplot as plt

path_to_wd = 'C:\\Users\\nagsa\\OneDrive\\Documents\\PhD_at_PennState\\Courses\\Spring2020\\CMPEN431_Computer_Architecture\\Project2\\CMPEN431_Project2'
os.chdir(path_to_wd)

WORKING_DIR = os.getcwd()
RESULTS_DIR = os.path.join(WORKING_DIR, 'results')
heuristic_folders = ['original_heuristic', 'our_heuristic']

metrics = ['performance', 'energy']
folder_name = 'logs'
file_names = ['ExecutionTime.best', 'ExecutionTime.log', 'EnergyEfficiency.best', 'EnergyEfficiency.log']
orig_path_performance_log = os.path.join(RESULTS_DIR, heuristic_folders[0], metrics[0], folder_name, file_names[1])
orig_path_performance_best =  os.path.join(RESULTS_DIR, heuristic_folders[0], metrics[0], folder_name, file_names[0])

orig_path_energy_log = os.path.join(RESULTS_DIR, heuristic_folders[0], metrics[1], folder_name, file_names[3])
orig_path_energy_best =  os.path.join(RESULTS_DIR, heuristic_folders[0], metrics[1], folder_name, file_names[2])

our_path_performance_log = os.path.join(RESULTS_DIR, heuristic_folders[1], metrics[0], folder_name, file_names[1])
our_path_performance_best =  os.path.join(RESULTS_DIR, heuristic_folders[1], metrics[0], folder_name, file_names[0])

our_path_energy_log = os.path.join(RESULTS_DIR, heuristic_folders[1], metrics[1], folder_name, file_names[3])
our_path_energy_best =  os.path.join(RESULTS_DIR, heuristic_folders[1], metrics[1], folder_name, file_names[2])

with open(orig_path_performance_log) as f:
    content = f.readlines() 

orig_perf_log = content

with open(orig_path_performance_best) as f:
    content = f.readlines() 

orig_perf_best = content

with open(orig_path_energy_log) as f:
    content = f.readlines() 

orig_energy_log = content

with open(orig_path_energy_best) as f:
    content = f.readlines() 

orig_energy_best = content


with open(our_path_performance_log) as f:
    content = f.readlines() 

our_perf_log = content

with open(our_path_performance_best) as f:
    content = f.readlines() 

our_perf_best = content

with open(our_path_energy_log) as f:
    content = f.readlines() 

our_energy_log = content

with open(our_path_energy_best) as f:
    content = f.readlines() 

our_energy_best = content

# Part A: line plots

# Original Performance
orig_perf_log_time = []
orig_perf_log_edp = []

for i in range(len(orig_perf_log)):
    temp = orig_perf_log[i].split(',')
    edp = float(temp[0])
    time = float(temp[1])
    
    orig_perf_log_time.append(time)
    orig_perf_log_edp.append(edp)
    
# Original Energy
orig_energy_log_time = []
orig_energy_log_edp = []

for i in range(len(orig_energy_log)):
    temp = orig_energy_log[i].split(',')
    edp = float(temp[0])
    time = float(temp[1])
    
    orig_energy_log_time.append(time)
    orig_energy_log_edp.append(edp)
    

# Our Performance
our_perf_log_time = []
our_perf_log_edp = []

for i in range(len(our_perf_log)):
    temp = our_perf_log[i].split(',')
    edp = float(temp[0])
    time = float(temp[1])
    
    our_perf_log_time.append(time)
    our_perf_log_edp.append(edp)
    
# Our Energy
our_energy_log_time = []
our_energy_log_edp = []

for i in range(len(our_energy_log)):
    temp = our_energy_log[i].split(',')
    edp = float(temp[0])
    time = float(temp[1])
    
    our_energy_log_time.append(time)
    our_energy_log_edp.append(edp)
    
orig_perf_time_steps = [x for x in range(len(orig_perf_log))]
orig_energy_time_steps =  [x for x in range(len(orig_energy_log))]

our_perf_time_steps = [x for x in range(len(our_perf_log))]
our_energy_time_steps =  [x for x in range(len(our_energy_log))]

'''

# Plot Ecexution Times
plt.figure()
ax = plt.gca()
ax.plot(orig_perf_time_steps, orig_perf_log_time, color = 'red', linewidth=3)
ax.yaxis.grid()
plt.title('NGET plot of Original Heuristic based on Performance')
plt.ylabel('Normalized Geomean Execution Time')
plt.xlabel('Iteration Number')
plt.savefig('nget_orig_perf.png')

plt.figure()
ax = plt.gca()
ax.plot(orig_energy_time_steps, orig_energy_log_time, color = 'blue', linewidth=3)
ax.yaxis.grid()
plt.title('NGET plot of Original Heuristic based on Energy')
plt.ylabel('Normalized Geomean Execution Time')
plt.xlabel('Iteration Number')
plt.savefig('nget_orig_energy.png')

plt.figure()
ax = plt.gca()
ax.plot(our_perf_time_steps, our_perf_log_time, color = 'red', linewidth=3)
ax.yaxis.grid()
plt.title('NGET plot of My Heuristic based on Performance')
plt.ylabel('Normalized Geomean Execution Time')
plt.xlabel('Iteration Number')
plt.savefig('nget_our_perf.png')

plt.figure()
ax = plt.gca()
ax.plot(our_energy_time_steps, our_energy_log_time, color = 'blue', linewidth=3)
ax.yaxis.grid()
plt.title('NGET plot of My Heuristic based on Energy')
plt.ylabel('Normalized Geomean Execution Time')
plt.xlabel('Iteration Number')
plt.savefig('nget_our_energy.png')


# Plot EDP
plt.figure()
ax = plt.gca()
ax.plot(orig_perf_time_steps, orig_perf_log_edp, color = 'red', linewidth=3)
ax.yaxis.grid()
plt.title('NGEDP plot of Original Heuristic based on Performance')
plt.ylabel('Normalized Geomean Energy Delay Product Time')
plt.xlabel('Iteration Number')
plt.savefig('ngedp_orig_perf.png')

plt.figure()
ax = plt.gca()
ax.plot(orig_energy_time_steps, orig_energy_log_edp, color = 'blue', linewidth=3)
ax.yaxis.grid()
plt.title('NGEDP plot of Original Heuristic based on Energy')
plt.ylabel('Normalized Geomean Energy Delay Product')
plt.xlabel('Iteration Number')
plt.savefig('ngedp_orig_energy.png')

plt.figure()
ax = plt.gca()
ax.plot(our_perf_time_steps, our_perf_log_edp, color = 'red', linewidth=3)
ax.yaxis.grid()
plt.title('NGEDP plot of My Heuristic based on Performance')
plt.ylabel('Normalized Geomean Energy Delay Product')
plt.xlabel('Iteration Number')
plt.savefig('ngedp_our_perf.png')

plt.figure()
ax = plt.gca()
ax.plot(our_energy_time_steps, our_energy_log_edp, color = 'blue', linewidth=3)
ax.yaxis.grid()
plt.title('NGEDP plot of My Heuristic based on Energy')
plt.ylabel('Normalized Geomean Energy Delay Product')
plt.xlabel('Iteration Number')
plt.savefig('ngedp_our_energy.png')

'''

orig_perf_best_line1 = orig_perf_best[0].split(',')
orig_perf_best_line2 = orig_perf_best[1].split(',')

orig_energy_best_line1 = orig_energy_best[0].split(',')
orig_energy_best_line2 = orig_energy_best[1].split(',')

our_perf_best_line1 = our_perf_best[0].split(',')
our_perf_best_line2 = our_perf_best[1].split(',')

our_energy_best_line1 = our_energy_best[0].split(',')
our_energy_best_line2 = our_energy_best[1].split(',')

'''
# bestTime = line2
# bestEDP = line1
plt.figure()
norm_time =  float(orig_perf_best_line2[2])
time0 = float(orig_perf_best_line2[6])
time1 = float(orig_perf_best_line2[8])
time2 = float(orig_perf_best_line2[10])
time3 = float(orig_perf_best_line2[12])
time4 = float(orig_perf_best_line2[14])


objects = ('B-0', 'B-1', 'B-2', 'B-3', 'B-4', 'GeoMean')
y_pos = np.arange(len(objects))
performance = [time0, time1, time2, time3, time4, norm_time]

plt.bar(y_pos, performance, align='center', color = 'red')
plt.xticks(y_pos, objects)
plt.ylabel('Normalized Execution Time')
plt.xlabel('Benchmarks')
plt.title('Bar Chart of Execution Times based on Performance of Original Heuristic')
plt.savefig('orig_perf_exectime.png')


plt.figure()
norm_time =  float(orig_energy_best_line2[2])
time0 = float(orig_energy_best_line2[6])
time1 = float(orig_energy_best_line2[8])
time2 = float(orig_energy_best_line2[10])
time3 = float(orig_energy_best_line2[12])
time4 = float(orig_energy_best_line2[14])


objects = ('B-0', 'B-1', 'B-2', 'B-3', 'B-4', 'GeoMean')
y_pos = np.arange(len(objects))
performance = [time0, time1, time2, time3, time4, norm_time]

plt.bar(y_pos, performance, align='center', color = 'blue')
plt.xticks(y_pos, objects)
plt.ylabel('Normalized Execution Time')
plt.xlabel('Benchmarks')
plt.title('Bar Chart of Execution Times based on Energy of Original Heuristic')
plt.savefig('orig_energy_exectime.png')




plt.figure()
norm_time =  float(our_perf_best_line2[2])
time0 = float(our_perf_best_line2[6])
time1 = float(our_perf_best_line2[8])
time2 = float(our_perf_best_line2[10])
time3 = float(our_perf_best_line2[12])
time4 = float(our_perf_best_line2[14])


objects = ('B-0', 'B-1', 'B-2', 'B-3', 'B-4', 'GeoMean')
y_pos = np.arange(len(objects))
performance = [time0, time1, time2, time3, time4, norm_time]

plt.bar(y_pos, performance, align='center', color = 'red')
plt.xticks(y_pos, objects)
plt.ylabel('Normalized Execution Time')
plt.xlabel('Benchmarks')
plt.title('Bar Chart of Execution Times based on Performance of My Heuristic')
plt.savefig('our_perf_exectime.png')


plt.figure()
norm_time =  float(our_energy_best_line2[2])
time0 = float(our_energy_best_line2[6])
time1 = float(our_energy_best_line2[8])
time2 = float(our_energy_best_line2[10])
time3 = float(our_energy_best_line2[12])
time4 = float(our_energy_best_line2[14])


objects = ('B-0', 'B-1', 'B-2', 'B-3', 'B-4', 'GeoMean')
y_pos = np.arange(len(objects))
performance = [time0, time1, time2, time3, time4, norm_time]

plt.bar(y_pos, performance, align='center', color = 'blue')
plt.xticks(y_pos, objects)
plt.ylabel('Normalized Execution Time')
plt.xlabel('Benchmarks')
plt.title('Bar Chart of Execution Times based on Energy of My Heuristic')
plt.savefig('our_energy_exectime.png')

'''
'''
plt.figure()
norm_time =  float(orig_perf_best_line1[1])
time0 = float(orig_perf_best_line1[6])
time1 = float(orig_perf_best_line1[8])
time2 = float(orig_perf_best_line1[10])
time3 = float(orig_perf_best_line1[12])
time4 = float(orig_perf_best_line1[14])


objects = ('B-0', 'B-1', 'B-2', 'B-3', 'B-4', 'GeoMean')
y_pos = np.arange(len(objects))
performance = [time0, time1, time2, time3, time4, norm_time]

plt.bar(y_pos, performance, align='center', color = 'red')
plt.xticks(y_pos, objects)
plt.ylabel('Normalized EDP')
plt.xlabel('Benchmarks')
plt.title('Bar Chart of EDP based on Performance of Original Heuristic')
plt.savefig('orig_perf_edp.png')


plt.figure()
norm_time =  float(orig_energy_best_line1[1])
time0 = float(orig_energy_best_line1[6])
time1 = float(orig_energy_best_line1[8])
time2 = float(orig_energy_best_line1[10])
time3 = float(orig_energy_best_line1[12])
time4 = float(orig_energy_best_line1[14])


objects = ('B-0', 'B-1', 'B-2', 'B-3', 'B-4', 'GeoMean')
y_pos = np.arange(len(objects))
performance = [time0, time1, time2, time3, time4, norm_time]

plt.bar(y_pos, performance, align='center', color = 'red')
plt.xticks(y_pos, objects)
plt.ylabel('Normalized EDP')
plt.xlabel('Benchmarks')
plt.title('Bar Chart of EDP based on Energy of Original Heuristic')
plt.savefig('orig_energy_edp.png')

plt.figure()
norm_time =  float(our_perf_best_line1[1])
time0 = float(our_perf_best_line1[6])
time1 = float(our_perf_best_line1[8])
time2 = float(our_perf_best_line1[10])
time3 = float(our_perf_best_line1[12])
time4 = float(our_perf_best_line1[14])


objects = ('B-0', 'B-1', 'B-2', 'B-3', 'B-4', 'GeoMean')
y_pos = np.arange(len(objects))
performance = [time0, time1, time2, time3, time4, norm_time]

plt.bar(y_pos, performance, align='center', color = 'red')
plt.xticks(y_pos, objects)
plt.ylabel('Normalized EDP')
plt.xlabel('Benchmarks')
plt.title('Bar Chart of EDP based on Performance of My Heuristic')
plt.savefig('our_perf_edp.png')

'''
plt.figure()
norm_time =  float(our_energy_best_line1[1])
time0 = float(our_energy_best_line1[6])
time1 = float(our_energy_best_line1[8])
time2 = float(our_energy_best_line1[10])
time3 = float(our_energy_best_line1[12])
time4 = float(our_energy_best_line1[14])


objects = ('B-0', 'B-1', 'B-2', 'B-3', 'B-4', 'GeoMean')
y_pos = np.arange(len(objects))
performance = [time0, time1, time2, time3, time4, norm_time]

plt.bar(y_pos, performance, align='center', color = 'blue')
plt.xticks(y_pos, objects)
plt.ylabel('Normalized EDP')
plt.xlabel('Benchmarks')
plt.title('Bar Chart of EDP based on Energy of My Heuristic')
plt.savefig('our_energy_edp.png')
