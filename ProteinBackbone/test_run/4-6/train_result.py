import os
import numpy as np
from matplotlib import pyplot as plt
from scipy import interpolate as intp
from scipy.interpolate import Rbf
from scipy.interpolate import UnivariateSpline as usp
from scipy.stats import linregress
from matplotlib import rcParams

f = plt.figure()
plt.rc('text', usetex=True) # use latex


def fileread(filename):
    """
    read .out file
    """
    epoch = []
    train_loss = []
    val_loss = []


    with open(filename, 'r') as file:
        while True:
            lines = file.readline()
            if not lines:
                break
            lines = lines.split()
            if (lines[0] == 'Epoch:') and ('Train' in lines):
                epoch_0 = int(lines[1])
                train_loss_0 = float(lines[5])
                #train_loss_0 = np.log10(float(lines[5]))
                epoch.append(epoch_0)  
                train_loss.append(train_loss_0)
            elif lines[0] == 'Evaluate':
                val_loss_0 = float(lines[3])
                val_loss.append(val_loss_0)

        epoch = np.array(epoch) 
        train_loss = np.array(train_loss)
        val_loss = np.array(val_loss)


    return epoch, train_loss, val_loss




def set_format():
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    rcParams['font.family']='sans-serif'
    rcParams['font.sans-serif']=['Arial'] 
    subplot=plt.gca()
    subplot.spines['bottom'].set_linewidth(0.5)
    subplot.spines['left'].set_linewidth(0.5)
    subplot.spines['right'].set_linewidth(0.5)
    subplot.spines['top'].set_linewidth(0.5)


def draw_spline(subplot, x, y, k):
    """
    draw spline (k: order)
    """
    data_list = []
    for i in range(0, len(x)):
        data_list.append((x[i], y[i]))
    def takeFirst(elem):
        return elem[0]
    data_list.sort(key = takeFirst)
    x = []
    y = []
    for i in range(0, len(data_list)):
        x.append(data_list[i][0])
        y.append(data_list[i][1])
    x = np.array(x) 
    y = np.array(y)
    f1 = usp(x, y, k = k)
    x_smooth = np.linspace(x.min(), x.max(), 3000)
    y_smooth = f1(x_smooth)   
    subplot.plot(x_smooth, y_smooth, zorder=1)
    set_format()




def draw_scatter(subplot, x, y, scatter_color='black'):
    """
    draw scatter
    """
    subplot.scatter(x, y, marker='.', color=scatter_color ,zorder=2)
    set_format()



def draw_plot_and_scatter(subplot, x, y, scatter_color='black', kind='linear', s=5):
    """
    draw plot and scatter
    """
    f1 = intp.interp1d(x, y, kind = kind)
    x_smooth = np.linspace(x.min(), x.max(), 3000)
    y_smooth = f1(x_smooth)   
    subplot.plot(x_smooth, y_smooth, zorder=1)
    subplot.scatter(x, y, marker='.', s=s, color=scatter_color, zorder=2)
    set_format()



def draw_spline_and_scatter(subplot, x, y, k, scatter_color='black', s=5):
    """
    draw spline and scatter
    """
    data_list = []
    for i in range(0, len(x)):
        data_list.append((x[i], y[i]))
    def takeFirst(elem):
        return elem[0]
    data_list.sort(key = takeFirst)
    x = []
    y = []
    for i in range(0, len(data_list)):
        x.append(data_list[i][0])
        y.append(data_list[i][1])
    x = np.array(x) 
    y = np.array(y)
    f1 = usp(x, y, k = k)
    x_smooth = np.linspace(x.min(), x.max(), 3000)
    y_smooth = f1(x_smooth)   
    subplot.plot(x_smooth, y_smooth, zorder=1)
    subplot.scatter(x, y, marker='.', color=scatter_color, zorder=2, s=s)
    set_format()

    
def draw_scatter_and_fit(subplot, x, y, scatter_color, s=5):
    """
    draw scatter and linear fit
    """
    slope, intercept, r_value, p_value, stderr = linregress(x, y)
    print(slope, intercept, r_value)
    xlims=[x[0], x[-1]]
    new_x = np.arange(xlims[0], xlims[1],(xlims[1]-xlims[0])/250.)
    subplot.scatter(x, y, marker='.', color=scatter_color, zorder=2, s=s)
    subplot.plot(new_x, intercept + slope *  new_x, zorder=1)
    set_format()




        
ax=f.add_subplot(111)
epoch, train_loss, val_loss = fileread('train_GNN.log')
plt.xlabel(r'$\rm epoch$')
# plt.ylabel(r'$\log_{10} ({\rm val\_loss})$')
plt.ylabel(r'$\log_{10} ({\rm train\_loss})$')
draw_plot_and_scatter(ax, epoch, train_loss)


f.set_size_inches(6,5)
plt.savefig('train-4-2-1.jpg',dpi=1000, bbox_inches='tight')
plt.show()

