import sys
import matplotlib.pyplot as plt
import csv
import numpy as np

def get_csv_data():
    """ Return the csv data from a csv file as a list with each element is a row of the csv file
        Helper function for hw5.
    """
    filename = sys.argv[1]
    csv_data = []
    with open(filename, encoding="utf-8") as f:
        reader = csv.reader(f)
        csv_data = list(reader)
    return csv_data

def process_csv(csv_data):
    """ Take a csv data list, then split into the header and data
        Helper function for hw5.
    """
    csv_header = csv_data[0]
    csv_row = csv_data[1:]
    return csv_header, csv_row

def get_year_day():
    """ Helper function for hw5. Get the years and days data from the csv file.
    """
    csv_header, csv_rows = process_csv(get_csv_data()) # call process csv to get csv rows and csv data
    year = [int(row[csv_header.index("year")]) for row in csv_rows]
    days = [int(row[csv_header.index("days")])for row in csv_rows]
    return year, days

def visualize():
    """ Function for question 2, get data and then save the plot of data in file "plot.jpg"
    """
    # call helper function to get data needed for plotting.
    year, ice_days = get_year_day()

    plt.plot(year, ice_days)
    plt.xlabel("Year")
    plt.ylabel("Number of frozen days")
    plt.savefig("plot.jpg")

def linear_regression():
    """ Function for question 3 -6. Based on given data, predict the best weight vectors for linear
    regression. Predict the number of days of lake froze based on the vector. Print out all answers
    """
    # call helper function to get data needed for calculation
    year, ice_days = get_year_day()
    # X array, features
    X = np.array([ [1, year[i]]for i in range(len(year))])
    print("Q3a:")
    print(X)

    # expected vectors (training data)
    y = np.array([ice_days[i] for i in range(len(ice_days))])
    print("Q3b:")
    print(y)

    # Z = X^TX
    Z = X.T @ X
    print("Q3c:")
    print(Z)

    # I = (X^TX)^-1
    I = np.linalg.inv(Z)
    print("Q3d:")
    print(I)

    # PI = (X^TX)^-1X^T 
    PI = I @ X.T
    print("Q3e:")
    print(PI)

    # beta^ = (X^TX)^-1X^Ty
    hat_beta = PI @ y
    print("Q3f:")
    print(hat_beta)
    
    # predict 2021's lake frozen days based on y^i = X_2021^TBeta
    y_test = hat_beta[0] + hat_beta[1] * 2021
    print("Q4: " + str(y_test))

    # calculate sign of beta_1
    sign = ""
    if hat_beta[1] > 0:
        sign = ">"
    elif hat_beta[1] < 0:
        sign = "<"
    else:
        sign = "="
    print("Q5a: " + sign)

    answer5b = """
    Since this is a linear model, each beta is the weight of its corresponding feature. Beta 1 is
    corresponding to the year. So the sign means the direction of change of lake mendota's days been convered
    in ice. The negative sign would mean as the year increases, the number of days will be decreasing. If the sign 
    is positive, then as the year increases, the number of days lake covered in ice will be increasing. It can also
    be intepreted as the slope of number of days covered in ice vs the year.
    """
    print("Q5b: " + answer5b)

    # 0 = beta[0] + beta[1]x* => x* = -beta[0]/beta[1]
    x_star = (hat_beta[0]*-1) / (hat_beta[1])
    print("Q6a: " + str(x_star))

    answer6b = """
    The prediction of x* is a not a compelling trend from this data. As the can be see from the trend of the data, the data has 
    a lot of variation each year. So it is not a perfect linear trend in that a linear model is not sufficient to predict the
    number of days lake will be covered in ice. We also only considering one feature (year) and there could be other feature in 
    the real world that affects the days of lake covered in ice. 
    """
    print("Q6b: " + answer6b)

def main():
    """ called when called main, execute all functions
    """
    visualize()
    linear_regression()

if __name__ == "__main__":
    main()