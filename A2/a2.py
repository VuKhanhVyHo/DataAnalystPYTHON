import math
import numpy as np 
import matplotlib.pyplot as plt

NUM_OF_HOUR = 24*365
NUM_OF_HALF_HOUR = 2*NUM_OF_HOUR

#PREDICT GRID DEMAND
def predict_demand(t_hours, a, p, b):
    """
    Calculates the grid demand D at time t_hours, for parameters a, p, b

    Parameters
    ----------
    t_hours: float
        Hours since 1 January 2017 00:00
    a : list[float]
        Demand (MW) corresponding to 12 h cycle, 24 h cycle, 365 d cycle and baseline demand
    p : list[float]
        Phase shift (radians) corresponding to 12 h cycle, 24 h cycle, 365 d cycle.
    b : float
        Rate of change in baseline demand (0.14 MW/d).

    Returns
    -------
    gd : float
        Predicted grid demand(MW) at time t_hours
    """
    gd = 0
    T = 12
    b /= 24
    for i in range (0,2):
        gd += a[i]*math.sin((2*math.pi*t_hours)/T + p[i])
        T+= 12
    gd += a[2]*math.sin((2*math.pi*t_hours)/(NUM_OF_HOUR + p[2]))
    gd +=  a[3] + b*t_hours
    return gd

#PREDICT HALF-HOURLY GRID DEMAND FOR 1 YEAR, USING LIST    
def demand_year_2(year, a, p, b):
    """ Predicts grid demand for year at half-hourly time steps, using lists

    Generates a list of 30 minute time intervals over year, and corresponding predictions of grid demand (MW) predicted using parameters a, p, b.
    If year is not in the range 2017 - 2019, the function runs for year 2017 and prints a statement indicating that the year 2017 is used

    Parameters
    ----------
    year: int
        Year for which demand is to be calculated.
    a : list[float]: Demand (MW) corresponding to 12 h, 24 h and 365 d cycles, and baseline demand
    p : list[float]: Phase shift (radians) corresponding to 12 h, 24 h and 365 d cycles.
    b: float: Rate of change in baseline demand (0.14MW/d).

    Returns
    -------
    h_list : list[float]: Hours since 1 January 00:00 for year, or 2017 if year out of range.
    gd_list: list[float]: Grid demand (MW) predicted for each element in h_list
    stats: tuple[float]: Mean, standard deviation and coefficient of variation of data in gd_list, all rounded to 2 decimal places
    year: int: year for which the functionn is run (2017 if year parameter is out of range) 
    """
    h_list_T = []
    gd_list = []
    stats = []
    year_fin = 0
    if year not in range (2017, 2020):
        print ("Year not in range! Default value of 2017 will be used")
        year_fin = 2017
    else: 
        year_fin = year
    hours = 0.0
    for i in range (0, NUM_OF_HALF_HOUR * 3):
        h_list_T.append(hours)
        hours += 0.5
    start = NUM_OF_HALF_HOUR*(year_fin - 2017)
    h_list = h_list_T[start:]
    for i in range(0, len(h_list)):
        gd_list.append(predict_demand(h_list[i], a, p, b))
    mean = 0
    for i in range(0, NUM_OF_HALF_HOUR):
        mean += gd_list[i]
    mean /= NUM_OF_HALF_HOUR
    std_dev = 0
    for i in range(0, NUM_OF_HALF_HOUR):
        std_dev += (gd_list[i] - mean)**2
    std_dev /= NUM_OF_HALF_HOUR
    std_dev = math.sqrt(std_dev)
    stats = (mean, std_dev, std_dev/mean)
    return h_list, gd_list, stats, year_fin

#PREDICT HALF-HOURLY GRID DEMAND FOR 1 YEAR USING ARRAYS
def demand_year_3(year, a, p, b):
    """
    Predicts grid demand for year at half - hourly time steps, using arrays

    Generates arrays of 30 minute time intervals over year, and corresponding predictions of grid demand (MW) using parameters a, p, b If year is not in the range 2017-2919, the function runs for year 2017 and prints a statement indicating the year 2017 is used

    Parameters
    ----------
    year : int
        Year for which demand is to be calculated
    a : list[float]
        Demand (MW) corresponding to 12 h, 24 h and 365 d cycles and baseline demand
    p: list[float]
        Phase shift  (radians) corresponding to 12h, 24h, and 365 d cycles.
    b: float
        Rate of change in baseline demand (0.14MW/d)

    Returns
    -------
    h_array : ndarray[float]
        Hours since 1 January 00:00 for year, or 2017 if year out of range.
    gd_array : ndarray[float]
        Grid demand (MW) predicted for each element in h_array.
    stats: tuple[float]
        Mean, standard deviation and coefficient of variation of data in gd_array. 
    year : int
        year for which the function is run (2017 if year parameter is out of range).
    """
    year_fin = 0
    if year not in range (2017, 2020):
        print ("Year not in range! Default value of 2017 will be used")
        year_fin = 2017
    else: 
        year_fin = year
    h_array = np.linspace(0.0, NUM_OF_HOUR*3 - 0.5, NUM_OF_HALF_HOUR*3)
    h_array = h_array[NUM_OF_HALF_HOUR*(year_fin - 2017):]
    gd_array = np.zeros_like(h_array)
    stats = np.arange(3, dtype = float)
    for i in range(0, len(h_array)):
        gd_array[i] = predict_demand(h_array[i], a, p, b)
    mean = 0
    for i in range(0, NUM_OF_HALF_HOUR):
        mean += gd_array[i]
    mean /= NUM_OF_HALF_HOUR
    stats[0] = mean
    std_dev = 0
    for i in range(0, NUM_OF_HALF_HOUR):
        std_dev += (gd_array[i] - mean)**2
    std_dev /= NUM_OF_HALF_HOUR
    std_dev = math.sqrt(std_dev)
    stats[1] = std_dev
    stats[2] = std_dev/mean
    return h_array, gd_array, stats, year_fin

#PLOT GRID DEMAND FROM ARRAYS
def plot_demand_byday(x, y, plot_title):
    """
    Plots grid demand for the first 28 days of data, and vs hours since midnight for all data

    Parameters
    ----------
    x : ndarray[floats]
        1d array of hours since 1 January 2017 00:00
    y : ndarray[floats]
        Grid demand (MW) at corresponding time in x for each element
    plot_title : str, optional.
        Super title of the figure. The default is ""
    Side-effects
    -------
    Generates two plots, stacked vertically.
    Top plot: Line plot of grid demand (MW) on y axis, vs time
              (hours since 1 January 2017 00:00) for first 28 days of data
    Lower plot: Scatter plot of grid demand (MW) on y axis,
              vs hours since midnight on x axis for all data
    Returns
    -------
    None.
    """
    x_day = (x[0:28*2*24] - x[0]) / 24 
    y_day = y[0:28*2*24]
    x_hour = (x - x[0]) % 24
    
    fig, axs = plt.subplots(2, 1)
    axs[0].plot(x_day, y_day)
    axs[1].scatter(x_hour, y)

    axs[0].set_xlabel("Days since start of simulation")
    axs[1].set_xlabel("Hours since midnight")

    fig.tight_layout()
    plt.suptitle(plot_title)
    fig.supylabel("Grid demand (MW)")
    plt.show()

#IMPORT DATA FROM FILE FOR COMPARISON WITH PREDICTIONS
def import_predict(fname, a, p, b):
    """ 
    Generates array with time and grid demand from file, predicted grid demand and residuals

    Parameters
    ----------
    fname : str
        Name of file to import.
    a : list[float]
        Demand (MW) corresponding to 12 h, 24 h, and 365 d cycles and baseline demand. 
    p : list[float]
        Phase shift (radians) corresponding to 12 h, 24 h, and 265 d cucles.
    b : float
        Rate of change in baseline demand (0.14 MW/d). 
    
    Returns 
    -------
    data_model: ndarray[float]: This array has four columns; The first column is
    hours since 1 January 2017 00:00, and the second column is grid demand (MW), 
    both imported from file fname. The third column is grid demand (MW), predicted
    at each corresponding time in column 1, using parameters a, p, b. The fourth 
    column contains the model residuals: observed - predicted grid demand (MW)
    """
    data = np.loadtxt(fname, skiprows = 1, delimiter =',')
    grid_predict = np.array([])
    for i in range (len(np.array(data[:,0]))):
        grid_predict = np.append(grid_predict, predict_demand(data[i,0], a, p, b))
    residual = np.array(data[:,1]) - grid_predict
    data_model = np.stack((np.array(data[:,0]), np.array(data[:,1]), grid_predict, residual), axis = 1)
    return data_model
    
#EVALUATE THE GRID DEMAND MODEL
def evaluate_model(data_model, plot_title):
    """
    Generates four subplots comparing model and predictions in data_model

    Parameters
    ----------
    data_model : ndarray[float]
        This array has four columns:
        The first column is hours since 1 January 2017 00:00.
        The second column is observed grid demand (MW),
        corresponding to each time in column 1.
        The third column is grid demand (MW), predicted at each corresponding time in column 1, using parameters a, p, b
        The fourth column contains the model residuals: observed - predicted grid demand (MW).

    Side-effects
    -------
    Generates four sub-plots: 
        Upper left: Scatter plot observed vs predicted grid demand, with 1:1 line
        Upper right: Boxplot of observed and predicted grid demand
        Lower left: Scatter plot of residuals vs hours since midnight
        Lower right: Histogram of residuals, with 10 bins
    
    Returns
    -------
    mstats: tuple[float]
        Mean and standard deviation of observed grid demand, and RMSE
        of model compared to data, all in MW and rounded to 2 decimal places
    """
    #CALCULATING
    mstats = np.array([])
    mean = np.mean(data_model[:,1])
    std_dev = np.std(data_model[:,1])
    rmse = 0.0
    for i in range(0, len(data_model)):
        rmse += (data_model[i,3])**2
    rmse /= len(data_model)
    rmse = np.sqrt(rmse)
    mstats =(round(mean,2), round(std_dev,2), round(rmse,2))

    #PLOT THE GRAPH
    x_hour = (data_model[:,0] - data_model[0][0]) % 24

    fig, axs = plt.subplots(2, 2)
    axs[0][0].scatter(data_model[:,1], data_model[:,2])
    axs[0][0].plot(data_model[:,1], data_model[:,1])
    axs[0][1].boxplot([data_model[:,1],data_model[:,2]])
    axs[1][0].scatter(x_hour, data_model[:,3])
    axs[1][1].hist(data_model[:,3])

    axs[0][0].set_xlabel("Observed demand (MW)")
    axs[0][1].set(xticklabels = ["Observed", "Predicted"])
    axs[1][0].set_xlabel("Hours since midnight")
    axs[1][1].set_xlabel("Residuals (MW)")


    axs[0][0].set_ylabel("Predicted demand (MW)")
    axs[0][1].set_ylabel("Demand (MW)")
    axs[1][0].set_ylabel("Residuals (MW)")
    axs[1][1].set_ylabel("Frequency")

    fig.tight_layout()
    plt.suptitle(plot_title)
    plt.show()

    return mstats


#REFINING THE GRID DEMAND MODEL
def compare_model(data_model, anew, pnew, bnew):
    """
    Runs grid demand model with new parameter values, and compare with gd0

    Parameters
    ----------
    data_model : ndarray[float]:
        This array has four columns: (time, observed grid demand, predicted grid
        demand and rediduals using the original parameters, as per output from \verb|import_predict|).
    anew : list[float]
        New demand (MW) parameters corresponding to 12 h, 24 h, and 365 d cycles and baseline demand.
    pnew : list[float]
        New values of phase shift (radians) corresponding to 12 h, 24 h, and 365 d cycles
    bnew : float 
        New rate of change in baseline demand (0.14 MW/d).
    
    Returns
    -------
    data_model_new : ndarray[float]:
        This array has four columns: The first column is hours since 1 January 
        2017 00:00. The second column is observed grid demand (MW),
        corresponding to each corresponding time in column 1, using
        parameters in Model1 (a1, p1, b1) The fourth column contains the model 
        residuals: observed - predicted grid demand (MW)
    rmses: list[float]
        RMSE of the original model, and of the revised model, rounded to 2 decimal 
        places, for the new model parameters.

    Side - effects
    ---------
        Generates plots by calling first plot_demand_byday, then evaluate_model.
        Prints a statement "The revised parameters produce better fit to the data:...",
        if RMSE of the revised parameters is less than the original RMSE, and 
        otherwise prints "The original parameters produce better fit to the data..."
        as per the sample output
    """
    rmse = 0.0
    for i in range(0, len(data_model)):
        rmse += (data_model[i,3])**2
    rmse /= len(data_model)
    rmse = np.sqrt(rmse)
    data_model_new = import_predict('ass2_qld.csv', anew, pnew, bnew)
    rmse_new = 0.0
    for i in range(0, len(data_model_new)):
        rmse_new += (data_model_new[i,3])**2
    rmse_new /= len(data_model_new)
    rmse_new = np.sqrt(rmse_new)
    print(rmse_new)
    if rmse_new < rmse:
        print("The revised parameters produce better fit to the data \n", rmse_new, " < ", rmse)
        plot_demand_byday(data_model_new[:,0], data_model_new[:,2], 'TEMPO')
    else:
        print("The original parameters produce better fit to the data")
    return data_model_new
A = [800,-400,200,6000]
phi = [0,0,0]
beta = 0.14
gd = predict_demand(125.5,A,phi, beta)
hours_2, demand_2, stats_2, year_2 = demand_year_3(2018, A,phi, beta)
data_model = import_predict('ass2_qld.csv', A, phi, beta)
A1 = [800, -400, 200, 6000]
phi1 = [-np.pi/2, -np.pi/2, np.pi/2]
data_model_new = compare_model(data_model, A1, phi1, beta)