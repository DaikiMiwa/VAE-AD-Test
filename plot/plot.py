# %%
import matplotlib.pyplot as plt
import pandas as pd

# %%
import matplotlib.pyplot as plt

def plot_power_graph(df, file_name, tag):
    if tag not in df["Tags"].unique():
        print(f"Tag '{tag}' not found in the dataframe.")
        return

    plt.rcParams["font.size"] = 16
    df = df.sort_values("signal", ascending=True)

    # Dynamically get unique x values if applicable
    x = [1,2,3,4]

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(x, df[df["Tags"] == tag]["rejection"], label="VAE-AD Test", marker='o', linestyle='-')
    ax.plot(x, df[df["Tags"] == tag]["oc-rejection"], label="OC", marker='o', linestyle='-')
    ax.plot(x, df[df["Tags"] == tag]["bonf-rejection"], label="Bonf", marker='o', linestyle='-')

    ax.set_xlabel("$\Delta$")
    ax.set_ylabel("Power")
    ax.set_xticks(x)
    ax.set_ylim([-0.05, 1.05])
    ax.legend(loc='upper left')
    ax.grid(True)
    # ax.set_title("Power Analysis")

    plt.savefig(file_name)

# %%
import matplotlib.pyplot as plt

def plot_error_rate_graph(df, file_name, tag):
    if tag not in df["Tags"].unique():
        print(f"Tag '{tag}' not found in the dataframe.")
        return

    plt.rcParams["font.size"] = 16
    df = df.sort_values("size", ascending=True)
    fig, ax = plt.subplots(figsize=(8, 6))

    # Assuming 'size' column exists and corresponds to the desired x-axis values
    x = [1,2,3,4]
    ax.set_xticks(x)

    ax.plot(x, df[df["Tags"] == tag]["rejection"], label="VAE-AD Test", marker='o', linestyle='-')
    ax.plot(x, df[df["Tags"] == tag]["oc-rejection"], label="OC", marker='o', linestyle='-')
    ax.plot(x, df[df["Tags"] == tag]["bonf-rejection"], label="Bonf", marker='o', linestyle='-')
    ax.plot(x, df[df["Tags"] == tag]["naive-rejection"], label="Naive", marker='o', linestyle='-')

    sizes = [64,256,1024,4096]
    ax.set_xticklabels(sizes)

    ax.set_xlabel("$n$")
    ax.set_ylabel("Type I Error Rate")
    ax.axhline(y=0.05,color='black', linestyle='--')
    # Add a dotted line from the first tick to the last tick on the x-axis
    # ax.plot([x[0], x[-1]], [0.05, 0.05], linestyle=':', color='black')
    ax.legend(loc='upper left')
    ax.grid(True)
    # ax.set_title("Type I Error Rate by Method")

    plt.savefig(file_name)


# %%
item_dict = {
    "submission_fpr": "submission_fpr",
    "submission_tpr": "submission_tpr",
    "submission_fpr_cov": "submission_fpr_cov",
    "submission_tpr_cov": "submission_tpr_cov"
}

# %%
file_name_list = [
    f"result_typeIerror.pdf",
    f"result_power.pdf",
    f"result_typeIerror_cov.pdf",
    f"result_power_cov.pdf"
]

# %%
df = pd.read_csv("result_synthetic.csv")

# %%
for file_name, tag in zip(file_name_list, item_dict.keys()):

    if "power" in file_name:
        plot_power_graph(df, file_name, tag)
    else :
        plot_error_rate_graph(df, file_name, tag)

# %%
import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
file_path = './result_robustness.csv'
data = pd.read_csv(file_path)

# Display the first few rows of the dataframe to understand its structure
data.head()

# %%


# %%
for alpha in [0.05,0.10]:
    plt.figure(figsize=(8, 6))
    plt.rcParams["font.size"] = 16
    file_path = './result_synthetic.csv'
    data = pd.read_csv(file_path)
    data = data[data["Tags"] == "submission_fpr_robust_after"]

    alpha_filtered_data = data[data['alpha'] == alpha]
    # alpha_filtered_data[alpha_filtered_data['noise_distribution']=='skewnorm']['number_of_iter']
    alpha_filtered_data[alpha_filtered_data['noise_distribution']=='skewnorm'].sort_values(by=['ws_distance'],ascending=True)

    ws_distances = [0.01,0.02,0.03,0.04]

    for disti in alpha_filtered_data['noise_distribution'].unique():
        plt.plot(ws_distances, alpha_filtered_data[alpha_filtered_data['noise_distribution']==disti].sort_values(by=['ws_distance'],ascending=True)['rejection'], label=disti, marker='o')

    # Labeling the axes
    plt.xlabel('WS Distance')
    plt.ylabel('Type I Error')
    plt.ylim(0.0,0.4)
    plt.xticks(ws_distances)

    # Adding a title
    # plt.title('Line Plot of Noise Distributions with Type I Error (α = 0.05)')

    # Adding a legend
    plt.legend()

    # Show plot
    plt.grid(True)
    plt.show()

# %%


# %%



