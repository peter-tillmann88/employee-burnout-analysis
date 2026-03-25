import os
import matplotlib.pyplot as plt
import seaborn as sns

from data_loading import load_data

OUTPUT_DIR = "../reports/figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

#histogram of burnouts 
def plot_burnout_distribution(df):
    plt.figure()
    sns.histplot(df["burn_rate"], bins=30)
    plt.title("Distribution of Burn Rate")
    plt.ylabel("Number of Employees")
    plt.savefig(f"{OUTPUT_DIR}/burnout_distribution.png")
    plt.close()

#burnout by amount of hours worked
def plot_burnout_vs_work_hours(df):
    plt.figure()
    sns.scatterplot(x="resource_allocation", y="burn_rate", data=df)
    plt.title("Burnout vs Work Hours")
    plt.savefig(f"{OUTPUT_DIR}/burnout_vs_work_hours.png")
    plt.close()

#burnout by job designation
def plot_burnout_by_designation(df):
    plt.figure()
    sns.barplot(x="designation", y="burn_rate", data=df)
    plt.title("Burnout by Job Level")
    plt.savefig(f"{OUTPUT_DIR}/burnout_by_designation.png")
    plt.close()

#WFH vs inoffice burnout
def plot_wfh_vs_burnout(df):
    plt.figure()
    sns.barplot(x="wfh_setup_available", y="burn_rate", data=df)
    plt.title("WFH vs Burnout")
    plt.savefig(f"{OUTPUT_DIR}/wfh_vs_burnout.png")
    plt.close()


def main():
    df = load_data()

    plot_burnout_distribution(df)
    plot_burnout_vs_work_hours(df)
    plot_burnout_by_designation(df)
    plot_wfh_vs_burnout(df)

    print("EDA plots saved")


if __name__ == "__main__":
    main()



